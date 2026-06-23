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

使用 RealModelRunner + Scheduler + BlockManager 跑完整推理。

### Task 4.1: 单条推理

```python
engine = LLMEngine(model_config, cache_config, device="cuda")
engine.model_runner = RealModelRunner("meta-llama/Llama-3.1-8B-Instruct")
outputs = engine.generate(
    prompts=[tokenizer.encode("What is AI?")],
    sampling_params=SamplingParams(max_tokens=50),
)
print(tokenizer.decode(outputs[0].output_token_ids))
```

### Task 4.2: 多请求并发推理

---

<a id="part-5-benchmark"></a>
## Part 5: Benchmark [进阶]

### Task 5.1: 对比 HF 默认 generate vs vkv-engine

```
测量：
  - Throughput (tokens/s)
  - Memory usage (peak GPU memory)
  - Max concurrent requests before OOM
```

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
# Part 1-3 需要 GPU
uv run pytest tests/test_phase6.py -k "part1" -v
uv run pytest tests/test_phase6.py -k "part2" -v

# 不需要 GPU 的测试用 mock
uv run pytest tests/test_phase6.py -k "cpu" -v
```
