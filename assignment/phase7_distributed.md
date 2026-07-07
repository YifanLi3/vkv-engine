# Phase 7: Distributed Inference

> **Est. time**: 10–15 hours | **Difficulty**: ★★★★★
> **Prerequisites**: Phase 6 done
> **Requires**: at least 2 GPUs (same node recommended)
> **New dependency**: `torch.distributed` (built into PyTorch)

---

## Table of Contents

- [Part 0: Background — why we need distributed inference](#part-0-background)
- [Part 1: DDP Worker Foundation (Level 1)](#part-1-ddp)
- [Part 2: Pipeline Parallelism (Level 2)](#part-2-pp)

---

<a id="part-0-background"></a>
## Part 0: Background — three paradigms of distributed inference

### Data Parallelism (DP)

Each GPU holds a **full copy of the model** and handles **different requests**.

```
GPU 0: [Full Model] ← requests 1, 2, 3
GPU 1: [Full Model] ← requests 4, 5, 6
```

**Use case**: model fits on a single card; you want more throughput.

### Pipeline Parallelism (PP)

**Different layers on different GPUs**; a request flows through them in order.

```
GPU 0: layers 0-10 ─→ GPU 1: layers 11-21
```

**Use case**: the model has many layers, single-GPU memory can't fit the whole thing, but a single layer fits.

### Tensor Parallelism (TP)

**Split each layer's weights** across GPUs; each card computes part in parallel. Requires all-reduce.

```
GPU 0: q_proj[0:half]  ─┐
                        ├─ all-reduce → merged output
GPU 1: q_proj[half:]   ─┘
```

**Use case**: a single layer's parameters are too big (e.g. 70B / 175B models).

Phase 7 implements DP (Level 1) and PP (Level 2). TP is left for Phase 8+.

---

<a id="part-1-ddp"></a>
## Part 1: DDP Worker Foundation [Level 1]

> **Files**: `vkv/engine/worker.py`, `examples/distributed_inference.py`
> **Tests**: `uv run pytest tests/test_phase7.py -k "part1" -v`

### Background

Phase 6's `examples/data_parallel.py` uses Python `multiprocessing` + `Queue` for DP.
It works, but production stacks (PyTorch/DeepSpeed/vLLM) all use `torch.distributed` + NCCL:

- **Performance**: NCCL communicates directly between GPU memories, much faster than CPU-mediated queues
- **Extensibility**: PP/TP later can reuse the same process group
- **Aligned with industry practice**: every serious distributed training/inference framework builds on this API

### Task 1.1: Implement the `Worker` class

**File**: `vkv/engine/worker.py`

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

### Task 1.2: Implement master–worker dispatch

**File**: `examples/distributed_inference.py`

Rank 0 acts as the master:
- Receives user requests
- Shards by rank
- Uses `dist.scatter` / `dist.broadcast` to send them to each worker

Other ranks act as workers:
- Receive their own requests
- Run local inference
- Use `dist.gather` to return results to rank 0

**Core API**:
```python
import torch.distributed as dist

dist.init_process_group("nccl", rank=rank, world_size=world_size)
dist.barrier()                    # global sync
dist.broadcast(tensor, src=0)     # broadcast from rank 0
dist.all_gather(...)              # gather from everyone
```

**Note**: `torch.distributed` transfers tensors, not raw strings. Two-step trick:
1. Master tokenizes → token id tensor
2. Broadcast tensor to workers
3. Workers infer, return tensors
4. Master decodes back to text

### Task 1.3: Launcher script

Start N workers with `torch.multiprocessing.spawn`:

```python
import torch.multiprocessing as mp

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(worker_main, args=(world_size, ...), nprocs=world_size, join=True)
```

---

<a id="part-2-pp"></a>
## Part 2: Pipeline Parallelism [Level 2]

> **Files**: `vkv/engine/pipeline_runner.py`, `examples/pipeline_inference.py`
> **Tests**: `uv run pytest tests/test_phase7.py -k "part2" -v`
> **Requires**: >= 2 GPUs

### Background

Part 1 is **DP** (each card runs a full model on different requests).
Part 2 is **PP** (different layers on different cards; a single request flows across cards).

**Why PP?**

- Model is bigger than one GPU's VRAM (e.g. Llama 70B needs 140 GB in FP16; an A100 has 80 GB)
- PP splits 32 layers across 2 GPUs: layers 0–15 → GPU 0, layers 16–31 → GPU 1
- Single request path: GPU 0 computes the first half → activation crosses to GPU 1 → GPU 1 computes the second half

### Core challenge: KV cache must follow the sharding

Each layer's attention reads/writes its own KV. If layer 5 lives on GPU 0, layer 5's KV must also be on GPU 0. Otherwise Q/K/V won't be on the same card at attention time → CUDA device mismatch.

Required changes:
1. **`BlockManager`** must support `layer_device_map: Dict[int, str]`, placing `gpu_key_cache[layer]` on the requested device.
2. **Model loading** must use `accelerate`'s `device_map` to manually shard layers; HF inserts hooks that move activations across GPUs automatically.
3. **`PagedCache.update`** needs no change — it goes through `block_manager.write_kv(block_id, layer, slot, K, V)`, so BlockManager decides which card the KV lives on.

### Task 2.1: Extend `BlockManager` to support per-layer devices

**File**: `vkv/engine/block_manager.py`

Add an optional parameter to `__init__`:

```python
def __init__(self, model_config, cache_config,
             device: str = "cpu",
             layer_device_map: Optional[Dict[int, str]] = None):
```

- If `layer_device_map is None`: all layers use `device` (backward compatible)
- If provided: `gpu_key_cache[i]` is created with `device=layer_device_map[i]`

**Hint**: only the `device` argument of `torch.zeros(...)` needs changing.

### Task 2.2: Implement `PipelineParallelRunner`

**File**: `vkv/engine/pipeline_runner.py`

Extend `RealModelRunner` or write from scratch. Key points:

```python
class PipelineParallelRunner(RealModelRunner):
    def __init__(self, model_name, block_manager, num_gpus, ...):
        # 1. Compute layer -> device map
        num_layers = cfg.num_hidden_layers
        layers_per_gpu = num_layers // num_gpus
        device_map = {
            "model.embed_tokens": 0,
            **{f"model.layers.{i}": (i // layers_per_gpu)
               for i in range(num_layers)},
            "model.norm": num_gpus - 1,
            "lm_head": num_gpus - 1,
        }

        # 2. Load model with device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        ).eval()

        # 3. block_manager must be constructed with a matching layer_device_map
```

### Task 2.3: End-to-end PP inference demo

**File**: `examples/pipeline_inference.py`

Runs in a single process (PP needs only one Python process, because HF uses hooks to move activations across cards — unlike TP which needs multi-process communication):

```python
runner = PipelineParallelRunner(
    model_name=MODEL_NAME,
    num_gpus=2,
    ...
)
output = runner.generate("What is AI?", max_new_tokens=30)
```

### Task 2.4: Observations

While running, watch with `nvidia-smi`:
- Both GPUs hold model weights (sharded)
- Both GPUs are busy during inference (pipeline of activations)

**Performance note**: single-request PP is **not faster** (may even be slower due to cross-GPU activation transfer), but it lets you fit larger models. Real speedup requires **many requests + micro-batching** (split each batch into micro-batches so different pipeline stages run concurrently). That's Phase 8 territory.
