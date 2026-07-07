# Phase 8: Continuous Batching — Real Concurrent Throughput

> **预计用时**: 15-20 小时 | **难度**: ★★★★★
> **前置知识**: Phase 6 完成
> **目标**: 让 `RealLLMEngine.step()` 一次 forward 处理多个 seq，真正提升多请求 throughput

---

## 目录

- [Part 0: Background — 为什么 Phase 6 的 vkv 反而慢](#part-0)
- [Part 1: BatchedPagedCache — 支持多 seq 的 KV cache](#part-1)
- [Part 2: Batched Decode — 单步处理多个 seq](#part-2)
- [Part 3: Batched Prefill — 处理不同长度的 prompt](#part-3)
- [Part 4: Benchmark — 对比新老实现](#part-4)

---

<a id="part-0"></a>
## Part 0: Background — 问题诊断

Phase 6 的 benchmark 显示：

```
Batch=1 → vkv 快 5.3x   ✓
Batch=8 → vkv 慢 3x     ✗
```

**根本原因**：`RealLLMEngine.step()` 里循环处理每个 seq，每个 seq 单独跑一次 forward：

```python
# 当前实现（Phase 6）
for seq in output.scheduled_seqs:
    logits, cache = self.model_runner.decode_step(last_token, paged_cache)  # ← 每个 seq 一次
    token_id = self.model_runner.sample(logits)
```

8 个请求 → 8 次 model.forward()，GPU 大部分时间在 launch kernel + 传输 KV，真正计算只用了一点点。

**vLLM 的做法**：一次 forward 处理所有活跃 seq，`batch_size = num_active_seqs`。这就是 Continuous Batching。

---

<a id="part-1"></a>
## Part 1: BatchedPagedCache [核心]

> **文件**: `vkv/engine/batched_paged_cache.py`
> **测试**: `uv run pytest tests/test_phase8.py -k "part1" -v`

### Background

原 `PagedCache` 是"一个请求一个 cache"，`update()` 期望 shape `[1, num_kv_heads, new_tokens, head_dim]`。

Batched decode 要求 `batch_size > 1`：`[batch, num_kv_heads, 1, head_dim]`。每个 batch 位置对应一个不同的 seq，各自有独立的 block_table 和 _seq_length。

### 关键设计变化

| | PagedCache (原) | BatchedPagedCache (新) |
|---|---|---|
| block_table | `List[int]` (单 seq) | `List[List[int]]` (每 seq 一份) |
| _seq_length | `int` | `List[int]` |
| K/V input shape | `[1, H, T, D]` | `[B, H, T, D]` |
| K/V output shape | `[1, H, seq_len, D]` | `[B, H, max_seq_len, D]` (padded) |

### Task 1.1: 实现 `BatchedPagedCache.__init__`

存 `batch_size`, `block_tables: List[List[int]]`, `_seq_lengths: List[int]`。

### Task 1.2: 实现 `BatchedPagedCache.update`

对每个 batch_idx 分别 write_kv 到自己的 block_table，然后 pad-gather 所有 seq。

### Task 1.3: 实现 `_pad_and_stack_kv`

各 seq 长度不同，需要 pad 到 `max_seq_len` 才能 stack 成 `[batch, ...]` tensor。

### Task 1.4: 实现 `get_mask_sizes` / `get_seq_length`

HF Cache 接口要求。用 `max(_seq_lengths)` 作为返回值。

---

<a id="part-2"></a>
## Part 2: Batched Decode [核心]

> **文件**: `vkv/engine/real_model_runner.py`, `vkv/engine/real_llm_engine.py`
> **测试**: `uv run pytest tests/test_phase8.py -k "part2" -v`

### Task 2.1: 新增 `RealModelRunner.batched_decode_step`

签名：
```python
def batched_decode_step(
    self,
    token_ids: List[int],           # 每个 seq 上一个 token
    batched_cache: BatchedPagedCache,
) -> Tuple[torch.Tensor, BatchedPagedCache]:
    """
    Returns:
      logits: [batch, vocab_size]
    """
```

内部：
1. 拼 input_tensor `[batch, 1]`
2. attention_mask `[batch, max_seq_len + 1]`（不同 seq 历史长度不同）
3. position_ids `[batch, 1]`：每个 seq 用自己的 `_seq_length`
4. model.forward
5. 返回 `output.logits[:, -1, :]`

### Task 2.2: 修改 `RealLLMEngine.step()` decode 分支

原来循环 decode 每个 seq，改成一次 batched decode：

```python
# 新实现
active_seqs = output.scheduled_seqs
batched_cache = self._get_or_build_batched_cache(active_seqs)
token_ids_input = [seq.token_ids[-1] for seq in active_seqs]

logits, batched_cache = self.model_runner.batched_decode_step(
    token_ids_input, batched_cache
)
new_tokens = [self.model_runner.sample(logits[i:i+1]) for i in range(len(active_seqs))]
```

### Task 2.3: 处理动态 batch（新 seq 加入 / 完成 seq 退出）

当有 seq 完成或有新 prefill 完成加入 running queue 时，`BatchedPagedCache` 需要更新（增删 batch_idx）。

---

<a id="part-3"></a>
## Part 3: Batched Prefill [进阶]

> **文件**: `vkv/engine/real_model_runner.py`
> **测试**: `uv run pytest tests/test_phase8.py -k "part3" -v`

### 挑战

不同 prompt 长度不同，需要 padding + attention_mask。或者用 packed representation（vLLM 的做法，把多个 seq 的 token 拼成一维，配 seq_lens 表示边界）。

### Task 3.1: `batched_prefill(prompts_list)` 
简化版：padding 到最长 prompt 长度，用 attention_mask 忽略 padding token。

---

<a id="part-4"></a>
## Part 4: Benchmark

> **文件**: `examples/benchmark.py`（扩展现有）

测量新老实现的 throughput：

| Batch | vkv (old, seq loop) | vkv (new, batched) | HF baseline |
|---|---|---|---|
| 1 | X | X' | Y |
| 8 | X | X' | Y |
| 16 | X | X' | Y |

**预期**：batched vkv 应该在 batch>=2 时超过 HF。

---

## 运行测试

```bash
uv run pytest tests/test_phase8.py -k "part1" -v      # CPU 也能跑
uv run pytest tests/test_phase8.py -k "part2" -v      # 需要 GPU
uv run pytest tests/test_phase8.py -k "part3" -v      # 需要 GPU
uv run python examples/benchmark.py                    # 扩展后的 benchmark
```
