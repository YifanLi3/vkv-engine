# Phase 6, Part 4: 端到端推理 [集成]

## 背景

Part 1-3 实现了：
- `PagedCache`：把 KV cache 存进 BlockManager 的分页内存池
- `RealModelRunner`：用真实 HuggingFace 模型做 prefill / decode / sample

Part 4 的目标：把 `RealModelRunner` 插进 `LLMEngine`，实现完整的生产级推理流程。

---

## 核心问题：MockModelRunner vs RealModelRunner

`LLMEngine.step()` 目前用 `MockModelRunner`：
- prefill：生成随机 KV 张量，手动写入 BlockManager
- decode：随机采样 token

用 `RealModelRunner` 后，KV 写入由 `PagedCache` 内部完成，引擎层只需：
- prefill：`runner.prefill(prompt_ids)` → 返回 `PagedCache`（KV 已写入）
- decode：`runner.decode_step(last_token_id, paged_cache)` → 返回 `(logits, paged_cache)`，再 sample

---

## Task 4.1: 单条推理

用 `RealModelRunner.generate()` 直接推理，不经过 LLMEngine scheduler。

**文件**: `examples/single_inference.py`

---

## Task 4.2: 多请求并发推理

创建 `RealLLMEngine`，继承 `LLMEngine`，重写 `step()` 用真实模型。

关键设计：
- 维护 `paged_caches: Dict[int, PagedCache]`，key 是 `seq_id`
- prefill 时：创建 PagedCache，调 `runner.prefill()`，存入字典
- decode 时：从字典取出 PagedCache，调 `runner.decode_step()`，sample token
- 序列完成时：调 `paged_cache.free()` 释放 block

**文件**: `examples/multi_inference.py`

---

## 运行测试

```bash
# Task 4.1
uv run python examples/single_inference.py

# Task 4.2
uv run python examples/multi_inference.py
```
