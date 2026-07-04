"""
Phase 7, Part 1: Worker — single-GPU worker for distributed inference.

Each Worker owns one GPU and holds a full RealLLMEngine instance.
Multiple Workers communicate via torch.distributed (NCCL backend) to
form a Data Parallel inference cluster.

Naming aligned with vLLM:
  vLLM:   Worker (each GPU process holds a WorkerBase)
  vkv:    Worker (this file)
"""

from typing import List

import torch
import torch.distributed as dist

from vkv.config import ModelConfig, CacheConfig
from vkv.engine.scheduler import SchedulerConfig
from vkv.sampling_params import SamplingParams


class Worker:
    """
    A single-GPU worker in a distributed inference setup.

    Each worker:
      1. Owns one GPU (identified by `rank`)
      2. Holds a full RealLLMEngine instance
      3. Communicates with other workers via torch.distributed (NCCL)

    Usage:
        worker = Worker(rank=0, world_size=2, model_name="...")
        outputs = worker.execute(prompts)
        worker.shutdown()
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        model_name: str,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig = None,
        master_addr: str = "127.0.0.1",
        master_port: str = "29500",
    ):
        """
        Initialize the worker: set device, join process group, load engine.

        TODO:
        1. Store self.rank, self.world_size
        2. Set torch.cuda.set_device(rank) so this process uses GPU `rank`
        3. Set os.environ["MASTER_ADDR"] and MASTER_PORT
        4. Call dist.init_process_group(
               backend="nccl", rank=rank, world_size=world_size
           )
        5. Load RealLLMEngine on device=f"cuda:{rank}"
        6. Store tokenizer reference: self.tokenizer = self.engine.model_runner.tokenizer
        """
        raise NotImplementedError("TODO: Implement Worker.__init__")

    def execute(
        self,
        prompts: List[str],
        max_tokens: int = 30,
    ) -> List[str]:
        """
        Run inference on a batch of prompts (this worker's shard).

        Args:
            prompts: List of text prompts assigned to this worker
            max_tokens: Max new tokens per prompt

        Returns:
            List of generated strings (same length as prompts)

        TODO:
        1. Tokenize each prompt to token IDs
        2. Call self.engine.generate(prompts=..., sampling_params=SamplingParams(max_tokens=...))
        3. Decode each output with self.tokenizer.decode(..., skip_special_tokens=True)
        4. Return list of strings
        """
        raise NotImplementedError("TODO: Implement Worker.execute")

    def barrier(self):
        """Synchronize all workers. Wraps dist.barrier()."""
        dist.barrier()

    def shutdown(self):
        """Clean up the process group.

        TODO: Call dist.destroy_process_group()
        """
        raise NotImplementedError("TODO: Implement Worker.shutdown")

    @property
    def is_master(self) -> bool:
        return self.rank == 0
