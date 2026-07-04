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
import torch.multiprocessing as mp

# Allow importing from vkv/ when spawned as a separate process
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vkv.config import ModelConfig, CacheConfig
from vkv.engine.scheduler import SchedulerConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 30


def worker_main(rank: int, world_size: int, prompts: List[str]):
    """
    Entry point for each spawned worker process.

    Rank 0 is the master. Currently every rank runs the same generation
    on its own shard (simple DP). Rank 0 also prints all results after
    collecting them via a barrier.

    TODO (Task 1.3):
    1. Create Worker(rank, world_size, MODEL_NAME, model_cfg, cache_cfg)
    2. Slice `prompts` into this worker's shard:
         shard = prompts[rank::world_size]
    3. Call worker.execute(shard, max_tokens=MAX_NEW_TOKENS)
    4. Print results with a rank prefix, e.g. f"[GPU {rank}] {output}"
    5. worker.barrier() before shutdown to prevent one process exiting
       while others still need collective ops
    6. worker.shutdown()
    """
    from vkv.engine.worker import Worker

    model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
    cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=300, num_cpu_blocks=20)
    scheduler_cfg = SchedulerConfig(max_num_seqs=8)

    # TODO: fill in the worker execution logic per docstring
    raise NotImplementedError("TODO: Implement worker_main")


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
