# Phase 6: Real Model Integration

> **预计用时**: 8-12 小时 | **难度**: ★★★★★  
> **前置知识**: Phase 1-2 完成  
> **需要 GPU**: 是（至少 16GB 显存，如 RTX 4090 / A100）  
> **需要**: `pip install transformers accelerate`

---

## 目录

- [Part 0: Background — 从 Mock 到真实模型](#part-0-background)
- [Part 1: HuggingFace 模型加载 (热身)](#part-1-loading)
- [Part 2: PagedCache — 自定义 HF Cache (核心)](#part-2-paged-cache)
- [Part 3: RealModelRunner — 替代 MockModelRunner (核心)](#part-3-model-runner)
- [Part 4: 端到端推理 (集成)](#part-4-e2e)
- [Part 5: Benchmark — 对比 HF 默认 vs vkv-engine (进阶)](#part-5-benchmark)

---

<a id="part-0-background"></a>
## Part 0: Background

到目前为止，我们用 MockModelRunner（随机 KV 数据）验证了调度和内存管理的正确性。
现在要对接真实模型，让 vkv-engine 真正跑推理。

集成路径（从 docs/model_integration_guide.md）：

```
Level 1: MockModelRunner ← 你已经做了（Phase 1-2）
Level 2: HuggingFace Hook ← 现在做这个
Level 3: Custom CUDA Kernel ← 未来优化
```

Level 2 的思路：实现 HuggingFace 的 Cache 接口，注入到 model.generate() 中，
让模型内部使用我们的 BlockManager 管理 KV cache。

---

<a id="part-1-loading"></a>
## Part 1: HuggingFace 模型加载 [热身]

> **文件**: `vkv/engine/real_model_runner.py`  
> **测试**: `uv run pytest tests/test_phase6.py -k "part1" -v`（需要 GPU）

### Task 1.1: 加载模型和 tokenizer

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
```

### Task 1.2: 提取模型配置 → ModelConfig

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
## Part 2: PagedCache — 自定义 HF Cache [核心]

> **文件**: `vkv/engine/paged_cache.py`  
> **测试**: `uv run pytest tests/test_phase6.py -k "part2" -v`

### Background

HuggingFace 的 generate() 接受 `past_key_values` 参数。
默认是 DynamicCache（torch.cat 方式）。
我们实现一个 PagedCache，内部用 BlockManager。

```python
# 注入方式：
outputs = model.generate(**inputs, past_key_values=paged_cache)
#                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                  传入我们的 cache 实例
```

### Task 2.1: 实现 `PagedCache` 继承 `transformers.Cache`
### Task 2.2: 实现 `PagedCache.update()` — 写入新 KV 到 BlockManager
### Task 2.3: 实现 `PagedCache.get_seq_length()` — 返回缓存长度

---

<a id="part-3-model-runner"></a>
## Part 3: RealModelRunner [核心]

> **文件**: `vkv/engine/real_model_runner.py`  
> **测试**: `uv run pytest tests/test_phase6.py -k "part3" -v`（需要 GPU）

### Task 3.1: 实现 `RealModelRunner.__init__()`

加载真实模型，初始化 BlockManager。

### Task 3.2: 实现 `RealModelRunner.prefill()`

跑真实的 prefill forward pass，把 KV 写入 BlockManager。

### Task 3.3: 实现 `RealModelRunner.decode_step()`

跑单 token decode，写入新 KV，返回 logits。

### Task 3.4: 实现 `RealModelRunner.sample()`

从 logits 中采样下一个 token（支持 temperature）。

---

<a id="part-4-e2e"></a>
## Part 4: 端到端推理 [集成]

> **文件**: `examples/single_inference.py`, `examples/multi_inference.py`  
> **测试**: `uv run pytest tests/test_phase6_part4.py -v`（需要 GPU）

### Background

Part 1-3 实现了 `RealModelRunner`，可以独立跑单条推理。
Part 4 把它插进 `LLMEngine`，实现多请求并发。

核心差异：`LLMEngine.step()` 目前用 `MockModelRunner`：
- prefill：生成随机 KV，手动写入 BlockManager
- decode：随机采样 token

换成 `RealModelRunner` 后，KV 写入由 `PagedCache` 内部完成，引擎层只需：
- prefill：`runner.prefill(prompt_ids)` → 返回 `PagedCache`（KV 已自动写入）
- decode：`runner.decode_step(last_token, paged_cache)` → `(logits, paged_cache)` → `sample(logits)`

### Task 4.1: 单条推理

直接用 `RealModelRunner.generate()` 跑推理，不经过 LLMEngine scheduler。

**文件**: `examples/single_inference.py`

```python
runner = RealModelRunner("TinyLlama/TinyLlama-1.1B-Chat-v1.0", block_manager, device="cuda")
output = runner.generate("What is AI?", max_new_tokens=50)
print(output)
```

需要填写的 TODO：
1. 初始化 `ModelConfig`（TinyLlama: num_layers=22, num_kv_heads=4, head_dim=64）
2. 初始化 `CacheConfig`（选择合适的 block_size 和 num_gpu_blocks）
3. 创建 `BlockManager` 和 `RealModelRunner`

### Task 4.2: 多请求并发推理

创建 `RealLLMEngine`，继承 `LLMEngine`，重写 `step()` 使用真实模型。

**文件**: `examples/multi_inference.py`

关键设计：
```python
class RealLLMEngine(LLMEngine):
    paged_caches: Dict[int, PagedCache]  # seq_id → PagedCache

    def step(self):
        output = self.scheduler.schedule()
        if output.is_prefill:
            for seq in output.scheduled_seqs:
                # prefill：runner 内部写 KV 进 BlockManager
                paged_cache = self.model_runner.prefill(seq 的 prompt token ids)
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
                self.paged_caches[seq.seq_id].free()  # 释放 block
                # 收集 RequestOutput
```

需要填写的 TODO：
1. `RealLLMEngine.__init__`：用 `RealModelRunner` 替换 `MockModelRunner`
2. `step()` prefill 分支：prefill 每个 seq，存入 `self.paged_caches`
3. `step()` decode 分支：decode 每个 seq，sample token，处理完成的 seq

### Task 4.3: 验证 block 无泄漏

每次 generate 完成后，`block_manager.stats.used_blocks` 应为 0。

```python
for _ in range(3):
    engine.generate(prompts=[...], sampling_params=sp)
    assert engine.block_manager.stats.used_blocks == 0
```

---

<a id="part-5-benchmark"></a>
## Part 5: Benchmark [进阶]

> **文件**: `examples/benchmark.py`  
> **运行**: `uv run python examples/benchmark.py`（需要 GPU）

### Background

对比两种 KV cache 方案的实际性能差异：

| | HF 默认 (DynamicCache) | vkv-engine (PagedCache) |
|---|---|---|
| 内存分配 | 每步 torch.cat，连续增长 | 预分配 block pool，按需取用 |
| 并发效率 | 每个请求独占显存 | 共享 block pool |
| 碎片化 | 高（序列长度不一时浪费） | 低（block 可复用） |

### Task 5.1: 单请求 Throughput 对比

**文件**: `examples/benchmark.py` → `benchmark_hf_default()` 和 `benchmark_vkv_engine()`

测量指标：
- Throughput：tokens/s（越高越好）
- Peak GPU memory：GB（越低越好）

**TODO**：`benchmark_hf_default()` 里调用 `model.generate()` 并返回生成的 token 数。

### Task 5.2: 最大并发请求数对比

**文件**: `examples/benchmark.py` → `benchmark_max_concurrent_hf()` 和 `benchmark_max_concurrent_vkv()`

逐步增加并发请求数，直到 OOM，记录最大值。

**TODO**：`benchmark_max_concurrent_hf()` 里用 batched input 测试 HF 并发上限。

### 预期结果

vkv-engine 在以下场景下优势明显：
- **多请求并发**：共享 block pool 减少碎片，支持更多并发
- **长序列**：不需要预先分配最大长度的连续显存

单请求 throughput 差距不大（主要瓶颈是模型计算本身）。

---

## 环境配置

```bash
# 添加依赖
uv add transformers accelerate

# 需要 HuggingFace token（访问 Llama 模型）
huggingface-cli login

# 或者用更小的公开模型测试
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

## 运行测试

```bash
# Part 1-3（需要 GPU）
uv run pytest tests/test_phase6.py -k "part1" -v
uv run pytest tests/test_phase6.py -k "part2" -v
uv run pytest tests/test_phase6.py -k "part3" -v

# Part 4（需要 GPU）
uv run pytest tests/test_phase6.py -k "part4" -v

# 不需要 GPU 的测试
uv run pytest tests/test_phase6.py -k "cpu" -v

# Part 5 Benchmark（需要 GPU）
uv run python examples/benchmark.py
```
