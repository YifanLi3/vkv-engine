# vkv-engine Architecture Overview

Reference guide for understanding what each module does and how they connect.
Useful for interview prep and codebase navigation.

---

## 1. Bird's-eye View — Call Graph

```
┌─────────────────────────────────────────────────────────────┐
│  User Entry Points (examples/)                              │
│    single_inference.py   → RealModelRunner.generate()       │
│    multi_inference.py    → RealLLMEngine.generate()         │
│    data_parallel.py      → multiprocess DP demo             │
│    distributed_inference.py → Worker (torch.distributed DP) │
│    benchmark.py          → HF vs vkv comparison             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
      ┌────────────────────────────────┐
      │  RealLLMEngine  (real_llm_engine.py)   │  top-level coordinator
      │    • add_request()             │
      │    • step()  ─────────────┐    │
      │    • generate()           │    │
      └────────────┬──────────────┘    │
                   │                   │
        ┌──────────┼───────────────────┴───────────────┐
        ▼          ▼                                   ▼
   ┌─────────┐  ┌─────────────────┐         ┌────────────────────┐
   │Scheduler│  │RealModelRunner   │────▶   │PagedCache          │
   │(sched.) │  │(real_model_      │(HF     │(paged_cache.py)    │
   │         │  │  runner.py)      │Cache)  │ update() / free()  │
   │schedule │  │ prefill/decode   │        └──────────┬─────────┘
   │postpr.  │  │ /sample          │                   │
   └────┬────┘  └──────────────────┘                   ▼
        │                                     ┌────────────────┐
        ▼                                     │ BlockManager   │
   ┌─────────┐                                │(block_manager) │
   │Sequence │──── owns block_table ─────────▶│  allocate/free │
   │(seq.py) │                                │  write_kv      │
   └─────────┘                                │  gather_kv     │
                                              └───────┬────────┘
                                                      │
                                                      ▼
                                             ┌────────────────┐
                                             │BlockAllocator  │
                                             │  free list     │
                                             └────────────────┘
```

---

## 2. Module Cheat Sheet

### Core Data Model

| Module | Class(es) | 一句话职责 |
|---|---|---|
| `config.py` | `ModelConfig`, `CacheConfig` | 全局配置（层数、KV heads、block size 等） |
| `sampling_params.py` | `SamplingParams` | 生成参数（temperature、max_tokens） |

### Memory Layer (Phase 1)

| Module | Class(es) | 一句话职责 |
|---|---|---|
| `block.py` | `Block` | 单个 block 的抽象（早期方案，每个 block 持有自己的张量） |
| `block_allocator.py` | `BlockAllocator` | Free list，管理哪些 block ID 空闲/已分配 |
| `block_manager.py` | `BlockManager` | 预分配的 KV 张量池 + block 生命周期管理，核心的 `write_kv`/`gather_kv` |
| `evictor.py` | `LRUEvictor` | LRU 淘汰策略（内存不够时踢出老 sequence） |
| `swapper.py` | `Swapper` | GPU ↔ CPU KV 换入换出 |
| `quantizer.py` | `PerTensor/PerChannel/GroupedQuantizer` | KV cache INT8/INT4 量化 |

### Request Layer (Phase 1-2)

| Module | Class(es) | 一句话职责 |
|---|---|---|
| `sequence.py` | `Sequence`, `SequenceStatus` | 一个请求的完整状态（token 列表、block table、prompt/output 划分） |
| `scheduler.py` | `Scheduler`, `SchedulerConfig`, `SchedulerOutput` | 决定"这一步跑哪些 seq、prefill 还是 decode、要不要 preempt" |

### Model Layer (Phase 6)

| Module | Class(es) | 一句话职责 |
|---|---|---|
| `model_runner.py` | `MockModelRunner` | Phase 1-2 用的假模型（产生随机 KV），测试用 |
| `real_model_runner.py` | `RealModelRunner` | 真 HuggingFace 模型的 wrapper：`prefill`/`decode_step`/`sample` |
| `paged_cache.py` | `PagedCache` | 继承 HF `Cache` 接口，把 KV 写入路由到 `BlockManager` |

### Engine Layer

| Module | Class(es) | 一句话职责 |
|---|---|---|
| `llm_engine.py` | `LLMEngine`, `RequestOutput` | 顶层协调者（scheduler + model_runner + block_manager），配 `MockModelRunner` |
| `real_llm_engine.py` | `RealLLMEngine` | 继承 `LLMEngine`，用 `RealModelRunner` 替换 mock，重写 `step()` |
| `monitor.py` | `Monitor`, `MetricsCollector` | 指标采集（throughput、block util、fragmentation）+ Prometheus 导出 |

### Distributed Layer (Phase 7)

| Module | Class(es) | 一句话职责 |
|---|---|---|
| `worker.py` | `Worker` | 单 GPU worker，`torch.distributed` 通信 + 每卡一份 `RealLLMEngine` |

---

## 3. 关键数据流

### 3.1 单请求 generate 全过程

```
user prompt "What is AI?"
      │
      │ tokenizer.encode()
      ▼
input_ids = [1, 1724, 338, ...]
      │
      ▼
runner.prefill(input_ids)
      │
      │ 1. new PagedCache
      │ 2. model.forward(input_ids, past_key_values=paged_cache)
      │    │
      │    │ 每层 attention 内部调用：
      │    │   paged_cache.update(K, V, layer_idx)
      │    │        │
      │    │        │ 1. 为每个 new token 算 (block_idx, slot_idx)
      │    │        │ 2. 需要新 block 时 → block_manager.allocate()
      │    │        │ 3. block_manager.write_kv(block_id, layer, slot, K, V)
      │    │        │ 4. block_manager.gather_kv(block_table, num_tokens, layer)
      │    │        ▼
      │    │      返回完整 KV 给 attention 计算
      │    ▼
      │  output.logits（形状 [1, seq_len, vocab_size]）
      ▼
loop for max_new_tokens:
      runner.decode_step(last_token, paged_cache)
      runner.sample(logits, temperature)
      │
      ▼
paged_cache.free()  → block_manager.free(block_ids)
      │
      ▼
tokenizer.decode(output_ids)
```

### 3.2 多请求 concurrent generate（RealLLMEngine.generate）

```
prompts = [p1, p2, p3]
      │
      │ for each: add_request → new Sequence → scheduler.add()
      ▼
loop while not scheduler.is_finished():
      output = scheduler.schedule()
              │
              │ 决定 is_prefill / decode，返回 scheduled_seqs
              ▼
      if output.is_prefill:
          for seq in scheduled_seqs:
              paged_cache = runner.prefill(seq.prompt_ids)
              self.paged_caches[seq.seq_id] = paged_cache
      else:  # decode
          for seq in scheduled_seqs:
              logits, cache = runner.decode_step(last_token, cache)
              token_id = runner.sample(logits)
              token_ids.append(token_id)
          finished = scheduler.postprocess(seqs, token_ids)
          for seq in finished:
              paged_cache.free()
              self.outputs[seq.seq_id] = RequestOutput(...)
```

---

## 4. 面试可能问的点 & 关键答案

### Q1: 为什么需要 paged KV cache？

**A**: 标准 HF `DynamicCache` 用 `torch.cat`，每次生成新 token 都要重新分配内存 → **碎片化严重**。特别是多请求时，每个请求预留连续显存，序列长度不一造成大量浪费。

Paged 方案借鉴 OS 虚拟内存：把 KV 切成固定大小的 block（比如 16 tokens/block），预分配 pool，动态分配。碎片率降到接近 0，显存利用率高，支持更多并发。

### Q2: KV cache 存的具体是什么？shape 是什么？

**A**: 每个 transformer layer 的 attention 里，K/V 矩阵是从 hidden states 算出来的。为了避免每次 decode 重算历史 token 的 K/V，把它们缓存起来。

`BlockManager.gpu_key_cache[layer_idx]` shape: `[num_gpu_blocks, num_kv_heads, block_size, head_dim]`

寻址：`gpu_key_cache[layer_idx][block_id, :, slot_idx, :]` 就是"第 layer 层、第 block_id 号 block、第 slot_idx 个 slot"的一个 token 的 K。

### Q3: PagedCache 怎么和 HuggingFace 集成？

**A**: HF 的 `model.forward(..., past_key_values=cache)` 里，attention 层会调 `cache.update(K, V, layer_idx)`。我们继承 `transformers.Cache` 基类，重写 `update()`：
1. 把新 token 的 K/V `write_kv` 进 BlockManager
2. `gather_kv` 把完整历史 K/V 拼起来返回给 attention

对 HF 模型完全透明，不改模型代码。

### Q4: prefill 和 decode 有什么区别？

| | Prefill | Decode |
|---|---|---|
| 输入 | 整个 prompt（比如 100 tokens） | 只有上一步生成的 1 个 token |
| KV cache | 一次性写入 prompt 的所有 KV | 追加 1 个 token 的 KV |
| Attention | Q shape `[1, H, 100, D]`（因果 mask） | Q shape `[1, H, 1, D]` |
| GPU 利用 | Compute-bound | Memory-bound |
| 频率 | 每请求 1 次 | 每请求 N 次（生成 N 个 token） |

### Q5: Scheduler 为什么要区分 prefill/decode？

**A**: 两阶段计算特性完全不同：
- prefill 一次处理很多 token，占用 GPU 计算充分 → 优先跑
- decode 每步只 1 token，多请求并发才能填满 GPU

标准做法：优先 prefill 直到 batch 满，然后进入 decode 循环。这是 **continuous batching** 的核心逻辑。

### Q6: 你的 vkv-engine 和 vLLM 相比缺什么？

**A**:
1. **PagedAttention CUDA kernel**：vLLM 的 attention kernel 直接从 block table 读 KV，不用 `gather_kv` 拼接。我们的实现是"拼完再算"，多一次 memcpy。
2. **Continuous batching 的真正 batched forward**：我们的 `step()` 循环每个 seq 单独跑 forward，vLLM 把所有 seq 的 token 拼进一个 batched forward。
3. **Tensor Parallelism**：只做了 DP（Level 1）和 PP（Level 2）。
4. **Prefix caching**：vLLM 支持自动检测共享前缀并复用 block（COW），我们只有基础的 fork API。

### Q7: 你的 DP 实现和 vLLM 的 Worker 有什么区别？

**A**:
- 相同：都是每 GPU 一进程 + `torch.distributed` NCCL
- 不同：我的 Worker 每卡完整模型独立处理请求，vLLM 的 Worker 每卡持有 sharded 模型（TP）+ 层分 GPU（PP），forward 每层可能 all-reduce

### Q8: 为什么用 `torch.distributed` 而不是 `multiprocessing.Queue`？

**A**:
- multiprocessing.Queue：CPU 中转，pickle 序列化，慢
- torch.distributed + NCCL：GPU 直连（NVLink/PCIe），几百 GB/s 带宽
- 后续加 TP/PP 时，NCCL 是必需的（每层都要 all-reduce）

---

## 5. 复习顺序建议

**从底向上读代码**：

1. `config.py` → 理解数据结构
2. `block.py` → `block_allocator.py` → `block_manager.py`（Phase 1 核心）
3. `sequence.py` → `scheduler.py`（Phase 2 核心）
4. `paged_cache.py` → `real_model_runner.py`（Phase 6 核心，最关键）
5. `llm_engine.py` → `real_llm_engine.py`（顶层协调）
6. `worker.py`（Phase 7）

**读 examples/ 里的入口**看整个流程怎么跑起来。
