"""
Phase 7, Part 1: Distributed Inference Entry Point

Launches N worker processes (one per GPU) using torch.multiprocessing.spawn.
Each worker joins the NCCL process group and runs its share of prompts.

Run:
    uv run python examples/distributed_inference.py

Requires at least 2 GPUs.
"""

import os
import sys
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Allow importing from vkv/ when spawned as a separate process
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vkv.config import ModelConfig, CacheConfig
from vkv.engine.scheduler import SchedulerConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 30


def worker_main(rank: int, world_size: int, prompts: List[str]):
    """
    Master-Worker entry point.

    - Rank 0 (master):
        holds original prompts, broadcasts to all ranks,
        gathers results, prints them.
    - Other ranks (workers):
        receive broadcasted prompts, run their shard, send results back.

    Uses `dist.broadcast_object_list` / `dist.gather_object` for
    Python object communication (internally uses pickle + NCCL).
    """
    from vkv.engine.worker import Worker

    model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
    cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=300, num_cpu_blocks=20)
    scheduler_cfg = SchedulerConfig(max_num_seqs=8)

    worker = Worker(
        rank,
        world_size,
        MODEL_NAME,
        model_cfg,
        cache_cfg,
        scheduler_cfg,
    )

    # --- 1. Broadcast prompts from rank 0 to all ranks ---
    # Only rank 0 holds real prompts; other ranks pass None as placeholder.
    obj_list = [prompts if rank == 0 else None]
    dist.broadcast_object_list(obj_list, src=0)
    all_prompts = obj_list[0]

    # --- 2. Each rank runs its own shard ---
    shard = all_prompts[rank::world_size]
    outputs = worker.execute(shard, max_tokens=MAX_NEW_TOKENS)

    # --- 3. Gather results back to rank 0 ---
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(outputs, gathered, dst=0)

    # --- 4. Only rank 0 prints ---
    if rank == 0:
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        for r, results in enumerate(gathered):
            r_shard = all_prompts[r::world_size]
            for prompt, output in zip(r_shard, results):
                print(f"[GPU {r}] {prompt}\n         -> {output}")

    worker.barrier()
    worker.shutdown()


def main():
    world_size = torch.cuda.device_count()
    print(f"Detected {world_size} GPU(s)")

    if world_size < 2:
        print("Warning: distributed inference expects >= 2 GPUs. "
              "Running with 1 GPU for smoke test.")

    prompts = [
        "What is artificial intelligence?",
        "Explain neural networks in one sentence.",
        "What is KV cache?",
        "Describe transformer models.",
        "How does self-attention work?",
        "What is model quantization?",
        "Explain paged attention.",
        "What is inference throughput?",
    ]

    mp.spawn(
        worker_main,
        args=(world_size, prompts),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
