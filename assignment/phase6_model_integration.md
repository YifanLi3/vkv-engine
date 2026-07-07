# Phase 6: Real Model Integration

> **Est. time**: 8–12 hours | **Difficulty**: ★★★★★
> **Prerequisites**: Phases 1–2 done
> **Requires GPU**: Yes (at least 16 GB VRAM, e.g. RTX 4090 / A100)
> **Dependencies**: `pip install transformers accelerate`

---

## Table of Contents

- [Part 0: Background — from mock to real model](#part-0-background)
- [Part 1: HuggingFace model loading (warm-up)](#part-1-loading)
- [Part 2: PagedCache — custom HF Cache (core)](#part-2-paged-cache)
- [Part 3: RealModelRunner — replaces MockModelRunner (core)](#part-3-model-runner)
- [Part 4: End-to-end inference (integration)](#part-4-e2e)
- [Part 5: Benchmark — HF default vs vkv-engine (advanced)](#part-5-benchmark)

---

<a id="part-0-background"></a>
## Part 0: Background

Until now we used `MockModelRunner` (random KV data) to verify scheduling and
memory management. Now we plug in a real model so vkv-engine can run actual inference.

Integration path (from `docs/model_integration_guide.md`):

```
Level 1: MockModelRunner ← already done (Phases 1–2)
Level 2: HuggingFace hook ← doing this now
Level 3: Custom CUDA kernel ← future optimization
```

Level 2 idea: implement HuggingFace's Cache interface and inject it into
`model.generate()`, so the model uses our BlockManager to manage the KV cache.

---

<a id="part-1-loading"></a>
## Part 1: HuggingFace model loading [warm-up]

> **File**: `vkv/engine/real_model_runner.py`
> **Tests**: `uv run pytest tests/test_phase6.py -k "part1" -v` (needs GPU)

### Task 1.1: Load the model and tokenizer

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
```

### Task 1.2: Extract model configuration → ModelConfig

```python
cfg = model.config
model_config = ModelConfig(
    num_layers=cfg.num_hidden_layers,
    num_kv_heads=cfg.num_key_value_heads,
    head_dim=cfg.hidden_size // cfg.num_attention_heads,
)
```

---

<a id="part-2-paged-cache"></a>
## Part 2: PagedCache — custom HF Cache [core]

> **File**: `vkv/engine/paged_cache.py`
> **Tests**: `uv run pytest tests/test_phase6.py -k "part2" -v`

### Background

HuggingFace's `generate()` accepts a `past_key_values` argument.
The default is `DynamicCache` (which uses `torch.cat`).
We implement a `PagedCache` backed by our `BlockManager`.

```python
# How it is injected:
outputs = model.generate(**inputs, past_key_values=paged_cache)
#                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                  pass our own cache instance
```

### Task 2.1: Implement `PagedCache`, inheriting `transformers.Cache`
### Task 2.2: Implement `PagedCache.update()` — write new KV into the BlockManager
### Task 2.3: Implement `PagedCache.get_seq_length()` — return current cached length

---

<a id="part-3-model-runner"></a>
## Part 3: RealModelRunner [core]

> **File**: `vkv/engine/real_model_runner.py`
> **Tests**: `uv run pytest tests/test_phase6.py -k "part3" -v` (needs GPU)

### Task 3.1: Implement `RealModelRunner.__init__()`

Load the real model and initialize the BlockManager.

### Task 3.2: Implement `RealModelRunner.prefill()`

Run a real prefill forward pass and write KV into the BlockManager.

### Task 3.3: Implement `RealModelRunner.decode_step()`

Run a single-token decode, write the new KV, and return logits.

### Task 3.4: Implement `RealModelRunner.sample()`

Sample the next token from logits (with temperature support).

---

<a id="part-4-e2e"></a>
## Part 4: End-to-end inference [integration]

> **Files**: `examples/single_inference.py`, `examples/multi_inference.py`
> **Tests**: `uv run pytest tests/test_phase6.py -k "part4" -v` (needs GPU)

### Background

Parts 1–3 built `RealModelRunner`, which can run single-sequence inference
on its own. Part 4 plugs it into `LLMEngine` for multi-request concurrency.

Core difference: `LLMEngine.step()` currently uses `MockModelRunner`:
- prefill: generate random KV and write it manually into BlockManager
- decode: sample a random token

With `RealModelRunner`, KV writes happen inside `PagedCache`, and the engine layer only needs to:
- prefill: `runner.prefill(prompt_ids)` → returns a `PagedCache` (KV already written)
- decode: `runner.decode_step(last_token, paged_cache)` → `(logits, paged_cache)` → `sample(logits)`

### Task 4.1: Single-request inference

Call `RealModelRunner.generate()` directly, bypassing the LLMEngine scheduler.

**File**: `examples/single_inference.py`

```python
runner = RealModelRunner("TinyLlama/TinyLlama-1.1B-Chat-v1.0", block_manager, device="cuda")
output = runner.generate("What is AI?", max_new_tokens=50)
print(output)
```

TODOs:
1. Initialize `ModelConfig` (TinyLlama: `num_layers=22, num_kv_heads=4, head_dim=64`)
2. Initialize `CacheConfig` (pick appropriate `block_size` and `num_gpu_blocks`)
3. Create `BlockManager` and `RealModelRunner`

### Task 4.2: Multi-request concurrent inference

Subclass `LLMEngine` as `RealLLMEngine` and override `step()` to use the real model.

**File**: `examples/multi_inference.py`

Design sketch:
```python
class RealLLMEngine(LLMEngine):
    paged_caches: Dict[int, PagedCache]  # seq_id → PagedCache

    def step(self):
        output = self.scheduler.schedule()
        if output.is_prefill:
            for seq in output.scheduled_seqs:
                # prefill: runner writes KV into BlockManager internally
                paged_cache = self.model_runner.prefill(seq's prompt token ids)
                self.paged_caches[seq.seq_id] = paged_cache
            return []
        else:
            token_ids = []
            for seq in output.scheduled_seqs:
                paged_cache = self.paged_caches[seq.seq_id]
                logits, paged_cache = self.model_runner.decode_step(
                    last_token_id, paged_cache
                )
                token_ids.append(self.model_runner.sample(logits))
            finished_seqs = self.scheduler.postprocess(seqs, token_ids)
            for seq in finished_seqs:
                self.paged_caches[seq.seq_id].free()  # release blocks
                # collect RequestOutput
```

TODOs:
1. `RealLLMEngine.__init__`: swap `MockModelRunner` for `RealModelRunner`
2. `step()` prefill branch: prefill each seq, store the PagedCache in `self.paged_caches`
3. `step()` decode branch: decode each seq, sample a token, handle finished seqs

### Task 4.3: Verify no block leaks

After each `generate()` completes, `block_manager.stats.used_blocks` should be 0.

```python
for _ in range(3):
    engine.generate(prompts=[...], sampling_params=sp)
    assert engine.block_manager.stats.used_blocks == 0
```

---

<a id="part-5-benchmark"></a>
## Part 5: Benchmark [advanced]

> **File**: `examples/benchmark.py`
> **Run**: `uv run python examples/benchmark.py` (needs GPU)

### Background

Compare the real-world performance of the two KV cache schemes:

| | HF default (`DynamicCache`) | vkv-engine (`PagedCache`) |
|---|---|---|
| Memory allocation | `torch.cat` per step, grows contiguously | Pre-allocated block pool, on-demand |
| Concurrency efficiency | Each request holds its own memory | Shared block pool |
| Fragmentation | High (waste when seq lengths differ) | Low (blocks are reusable) |

### Task 5.1: Single-request throughput comparison

**File**: `examples/benchmark.py` → `benchmark_hf_default()` and `benchmark_vkv_engine()`

Metrics:
- Throughput (tokens/s, higher is better)
- Peak GPU memory (GB, lower is better)

**TODO**: in `benchmark_hf_default()`, call `model.generate()` and return the number of generated tokens.

### Task 5.2: Maximum concurrent-request comparison

**File**: `examples/benchmark.py` → `benchmark_max_concurrent_hf()` and `benchmark_max_concurrent_vkv()`

Gradually increase concurrent requests until OOM; record the maximum.

**TODO**: in `benchmark_max_concurrent_hf()`, use batched inputs to find HF's concurrency ceiling.

### Expected results

vkv-engine wins clearly under:
- **Multi-request concurrency** — shared block pool cuts fragmentation, supporting more concurrent seqs
- **Long sequences** — no need to pre-allocate maximum-length contiguous memory

For single-request throughput the difference is small (compute cost dominates).

---

## Environment setup

```bash
# Add dependencies
uv add transformers accelerate

# HuggingFace token needed (for Llama access)
huggingface-cli login

# Or use a smaller open model for testing:
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

## Running the tests

```bash
# Parts 1–3 (need GPU)
uv run pytest tests/test_phase6.py -k "part1" -v
uv run pytest tests/test_phase6.py -k "part2" -v
uv run pytest tests/test_phase6.py -k "part3" -v

# Part 4 (needs GPU)
uv run pytest tests/test_phase6.py -k "part4" -v

# CPU-only tests
uv run pytest tests/test_phase6.py -k "cpu" -v

# Part 5 benchmark (needs GPU)
uv run python examples/benchmark.py
```
