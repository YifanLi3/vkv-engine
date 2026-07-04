# Phase 7: Distributed Inference

> **预计用时**: 10-15 小时 | **难度**: ★★★★★
> **前置知识**: Phase 6 完成
> **需要**: 至少 2 张 GPU（推荐同一节点）
> **新增依赖**: `torch.distributed`（PyTorch 内置）

---

## 目录

- [Part 0: Background — 为什么需要分布式推理](#part-0-background)
- [Part 1: DDP Worker Foundation (Level 1)](#part-1-ddp)
- [Part 2: Pipeline Parallelism (Level 2)](#part-2-pp)

---

<a id="part-0-background"></a>
## Part 0: Background — 分布式推理的三种范式

### Data Parallelism (DP)

每张 GPU 一份**完整模型**，处理**不同的请求**。

```
GPU 0: [Full Model] ← request 1, 2, 3
GPU 1: [Full Model] ← request 4, 5, 6
```

**适用场景**：模型能装单卡，想提高 throughput。

### Pipeline Parallelism (PP)

**不同的层放到不同 GPU**，一个请求依次流过。

```
GPU 0: layer 0-10 ─→ GPU 1: layer 11-21
```

**适用场景**：模型层数多，单卡装不下但单层能装下。

### Tensor Parallelism (TP)

**每一层的权重切分**到多卡，各卡并行算一部分。需要 all-reduce。

```
GPU 0: q_proj[0:half]  ─┐
                        ├─ all-reduce → 合并输出
GPU 1: q_proj[half:]   ─┘
```

**适用场景**：单层参数太大（如 70B/175B 模型）。

Phase 7 实现 DP (Level 1) 和 PP (Level 2)。TP 留给 Phase 8+。

---

<a id="part-1-ddp"></a>
## Part 1: DDP Worker Foundation [Level 1]

> **文件**: `vkv/engine/worker.py`, `examples/distributed_inference.py`
> **测试**: `uv run pytest tests/test_phase7.py -k "part1" -v`

### Background

Phase 6 的 `examples/data_parallel.py` 用 Python multiprocessing + Queue 做 DP，够用但不规范。工业界（PyTorch/DeepSpeed/vLLM）都用 `torch.distributed` + NCCL：

- **性能**：NCCL 直接在 GPU 内存间通信，比 CPU 中转的 Queue 快
- **可扩展**：将来加 PP/TP 时直接复用 process group
- **对齐工业实践**：所有分布式训练/推理框架都基于这套 API

### Task 1.1: 实现 `Worker` 类

**文件**: `vkv/engine/worker.py`

```python
class Worker:
    """
    A single-GPU worker in a distributed inference setup.

    Each worker:
    1. Owns one GPU (based on rank)
    2. Holds a full RealLLMEngine instance
    3. Communicates with other workers via torch.distributed (NCCL)
    """

    def __init__(self, rank: int, world_size: int, model_name: str, ...):
        # TODO:
        # 1. Store rank, world_size
        # 2. Set CUDA device to `rank`
        # 3. Initialize process group with NCCL backend
        # 4. Load a full RealLLMEngine on this GPU
        pass

    def execute(self, prompts: List[str]) -> List[str]:
        """Process a batch of prompts on this worker's GPU."""
        # TODO: run engine.generate() and return decoded outputs
        pass

    def shutdown(self):
        """Clean up the process group."""
        # TODO: dist.destroy_process_group()
        pass
```

### Task 1.2: 实现 Master-Worker 分发

**文件**: `examples/distributed_inference.py`

Rank 0 作为 master：
- 接收用户请求
- 按 rank 分片
- 用 `dist.scatter` / `dist.broadcast` 发给各 worker

其他 rank 作为 worker：
- 接收自己的请求
- 本地推理
- 用 `dist.gather` 把结果返回 rank 0

**核心 API**:
```python
import torch.distributed as dist

dist.init_process_group("nccl", rank=rank, world_size=world_size)
dist.barrier()          # 全员同步
dist.broadcast(tensor, src=0)   # 从 rank 0 广播
dist.all_gather(...)    # 全员收集
```

**注意**：`torch.distributed` 只能传 tensor，不能直接传字符串。需要用两步：
1. 主进程 tokenize，得到 token id tensor
2. broadcast tensor 到各 worker
3. worker 推理返回 tensor
4. 主进程 decode 回文字

### Task 1.3: 启动脚本

用 `torch.multiprocessing.spawn` 启动 N 个 worker：

```python
import torch.multiprocessing as mp

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(worker_main, args=(world_size, ...), nprocs=world_size, join=True)
```

---

<a id="part-2-pp"></a>
## Part 2: Pipeline Parallelism [Level 2]

> Part 1 完成后再展开细节。

**概览**：
- 用 `accelerate` 的 `device_map` 手动把不同层放到不同 GPU
- 前向传播时 activations 自动跨 GPU 传递
- 挑战：KV cache 每层在不同 GPU 上，需要修改 `BlockManager` 感知设备
