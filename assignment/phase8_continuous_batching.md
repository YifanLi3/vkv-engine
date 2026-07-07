# Phase 8: Continuous Batching — Real Concurrent Throughput

> **Est. time**: 15–20 hours | **Difficulty**: ★★★★★
> **Prerequisites**: Phase 6 done
> **Goal**: make `RealLLMEngine.step()` process multiple sequences in a single forward pass, so multi-request throughput actually improves

---

## Table of Contents

- [Part 0: Background — why Phase 6's vkv is slower under load](#part-0)
- [Part 1: BatchedPagedCache — KV cache for multiple sequences](#part-1)
- [Part 2: Batched decode — process many seqs per step](#part-2)
- [Part 3: Batched prefill — handle prompts of different lengths](#part-3)
- [Part 4: Benchmark — compare old vs new](#part-4)

---

<a id="part-0"></a>
## Part 0: Background — problem diagnosis

Phase 6's benchmark showed:

```
Batch=1 → vkv 5.3x faster    ✓
Batch=8 → vkv 3x slower      ✗
```

**Root cause**: `RealLLMEngine.step()` iterates over each seq and runs one forward per seq:

```python
# Current implementation (Phase 6)
for seq in output.scheduled_seqs:
    logits, cache = self.model_runner.decode_step(last_token, paged_cache)  # ← one forward per seq
    token_id = self.model_runner.sample(logits)
```

8 requests → 8 `model.forward()` calls. The GPU spends most of its time launching kernels and moving KV around, doing very little real compute.

**vLLM's approach**: one forward per step for **all** active seqs, `batch_size = num_active_seqs`. That's Continuous Batching.

---

<a id="part-1"></a>
## Part 1: BatchedPagedCache [core]

> **File**: `vkv/engine/batched_paged_cache.py`
> **Tests**: `uv run pytest tests/test_phase8.py -k "part1" -v`

### Background

The original `PagedCache` was "one cache per request"; `update()` expects shape `[1, num_kv_heads, new_tokens, head_dim]`.

Batched decode needs `batch_size > 1`: `[batch, num_kv_heads, 1, head_dim]`. Each batch index corresponds to a different seq, each with its own `block_table` and `_seq_length`.

### Key design differences

| | PagedCache (old) | BatchedPagedCache (new) |
|---|---|---|
| block_table | `List[int]` (single seq) | `List[List[int]]` (one per seq) |
| _seq_length | `int` | `List[int]` |
| K/V input shape | `[1, H, T, D]` | `[B, H, T, D]` |
| K/V output shape | `[1, H, seq_len, D]` | `[B, H, max_seq_len, D]` (padded) |

### Task 1.1: Implement `BatchedPagedCache.__init__`

Store `batch_size`, `block_tables: List[List[int]]`, `_seq_lengths: List[int]`.

### Task 1.2: Implement `BatchedPagedCache.update`

For each `batch_idx`, `write_kv` into that seq's own block_table; then pad-gather across all seqs.

### Task 1.3: Implement `_pad_and_stack_kv`

Sequences have different lengths, so pad to `max_seq_len` before stacking into a `[batch, ...]` tensor.

### Task 1.4: Implement `get_mask_sizes` / `get_seq_length`

Required by the HF Cache interface. Return `max(_seq_lengths)`.

---

<a id="part-2"></a>
## Part 2: Batched decode [core]

> **Files**: `vkv/engine/real_model_runner.py`, `vkv/engine/real_llm_engine.py`
> **Tests**: `uv run pytest tests/test_phase8.py -k "part2" -v`

### Task 2.1: Add `RealModelRunner.batched_decode_step`

Signature:
```python
def batched_decode_step(
    self,
    token_ids: List[int],           # last token of each seq
    batched_cache: BatchedPagedCache,
) -> Tuple[torch.Tensor, BatchedPagedCache]:
    """
    Returns:
      logits: [batch, vocab_size]
    """
```

Inside:
1. Build input_tensor `[batch, 1]`
2. attention_mask `[batch, max_seq_len + 1]` (seqs have different histories)
3. position_ids `[batch, 1]` — each seq uses its own `_seq_length`
4. `model.forward`
5. Return `output.logits[:, -1, :]`

### Task 2.2: Modify `RealLLMEngine.step()` decode branch

Replace the per-seq loop with a single batched decode:

```python
# New implementation
active_seqs = output.scheduled_seqs
batched_cache = self._get_or_build_batched_cache(active_seqs)
token_ids_input = [seq.token_ids[-1] for seq in active_seqs]

logits, batched_cache = self.model_runner.batched_decode_step(
    token_ids_input, batched_cache
)
new_tokens = [self.model_runner.sample(logits[i:i+1]) for i in range(len(active_seqs))]
```

### Task 2.3: Handle dynamic batches (seqs joining / leaving)

When a seq finishes or a newly-prefilled seq joins the running queue, `BatchedPagedCache` must be updated (add/remove `batch_idx`).

---

<a id="part-3"></a>
## Part 3: Batched prefill [advanced]

> **File**: `vkv/engine/real_model_runner.py`
> **Tests**: `uv run pytest tests/test_phase8.py -k "part3" -v`

### Challenge

Prompts have different lengths → need padding + attention_mask. Alternative: packed representation (vLLM's approach: concatenate all seqs' tokens into a 1-D tensor with `seq_lens` marking boundaries).

### Task 3.1: `batched_prefill(prompts_list)`

Simplified version: pad to the longest prompt, use `attention_mask` to ignore padding tokens.

---

<a id="part-4"></a>
## Part 4: Benchmark

> **File**: `examples/benchmark.py` (extend existing)

Measure old vs new throughput:

| Batch | vkv (old, seq loop) | vkv (new, batched) | HF baseline |
|-------|---------------------|--------------------|-------------|
| 1     | X                   | X'                 | Y           |
| 8     | X                   | X'                 | Y           |
| 16    | X                   | X'                 | Y           |

**Expected**: batched vkv should surpass HF once `batch >= 2`.

---

## Running the tests

```bash
uv run pytest tests/test_phase8.py -k "part1" -v      # CPU-compatible
uv run pytest tests/test_phase8.py -k "part2" -v      # needs GPU
uv run pytest tests/test_phase8.py -k "part3" -v      # needs GPU
uv run python examples/benchmark.py                    # extended benchmark
```
