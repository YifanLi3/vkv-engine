"""
Phase 7 Tests: Distributed Inference

Part 1: DDP Worker Foundation (Level 1)
Part 2: Pipeline Parallelism (Level 2)  — TBD

Requires >= 2 GPUs for real distributed tests.
Single-GPU smoke tests are also included.
"""

import os
import sys
import pytest
import torch
import torch.multiprocessing as mp

from vkv.config import ModelConfig, CacheConfig
from vkv.engine.scheduler import SchedulerConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

has_two_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Requires at least 2 GPUs",
)
has_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires a CUDA GPU",
)


# ─────────────────────────────────────────────────────────
# Helpers for spawning workers in tests
# ─────────────────────────────────────────────────────────

def _worker_smoke(rank, world_size, result_queue):
    """Init worker, run one prompt, put output in queue, shutdown."""
    from vkv.engine.worker import Worker

    model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
    cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=200, num_cpu_blocks=20)
    scheduler_cfg = SchedulerConfig(max_num_seqs=4)

    worker = Worker(
        rank=rank,
        world_size=world_size,
        model_name=MODEL_NAME,
        model_config=model_cfg,
        cache_config=cache_cfg,
        scheduler_config=scheduler_cfg,
    )
    outputs = worker.execute(["Hello"], max_tokens=5)
    result_queue.put((rank, outputs))
    worker.barrier()
    worker.shutdown()


# ─────────────────────────────────────────────────────────
# Part 1: DDP Worker Tests
# ─────────────────────────────────────────────────────────

class TestPart1:

    @has_two_gpus
    def test_worker_init_and_shutdown(self):
        """Spawn 2 workers, they should init NCCL and shut down cleanly."""
        world_size = 2
        ctx = mp.get_context("spawn")
        q = ctx.Queue()

        procs = [
            ctx.Process(target=_worker_smoke, args=(r, world_size, q))
            for r in range(world_size)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=180)

        assert all(p.exitcode == 0 for p in procs), \
            f"Worker processes failed: {[p.exitcode for p in procs]}"

    @has_two_gpus
    def test_worker_execute_returns_strings(self):
        """Each worker should produce a non-empty output string."""
        world_size = 2
        ctx = mp.get_context("spawn")
        q = ctx.Queue()

        procs = [
            ctx.Process(target=_worker_smoke, args=(r, world_size, q))
            for r in range(world_size)
        ]
        for p in procs:
            p.start()

        results = [q.get(timeout=180) for _ in range(world_size)]

        for p in procs:
            p.join(timeout=30)

        # Each rank returned exactly one output
        assert len(results) == world_size
        for rank, outputs in results:
            assert len(outputs) == 1
            assert isinstance(outputs[0], str)
            assert len(outputs[0]) > 0
