"""
Phase 1, Part 7: GPU ↔ CPU Swapper

nano-vLLM doesn't implement swap — it just drops preempted sequences
and re-prefills them later. vLLM has full swap support.

Our Swapper preserves KV cache in CPU memory during preemption,
avoiding expensive re-computation when the sequence resumes.
This is one of the areas where vkv-engine goes deeper than nano-vLLM.
"""

from typing import Dict, List

import torch

from vkv.engine.block_manager import BlockManager


class Swapper:
    """
    Manages GPU ↔ CPU block transfers.

    Uses the BlockManager's pre-allocated GPU and CPU cache tensors.

    Flow:
        swap_out: GPU block → copy data to CPU block → free GPU block
        swap_in:  CPU block → copy data to GPU block → free CPU block
    """

    def __init__(self, block_manager: BlockManager):
        """
        Initialize the swapper.

        Args:
            block_manager: BlockManager that owns both GPU and CPU memory

        TODO: Implement this.
        Store the block_manager reference.
        """
        raise NotImplementedError("TODO: Implement Swapper.__init__")

    def swap_out(self, gpu_block_ids: List[int]) -> Dict[int, int]:
        """
        Copy blocks from GPU to CPU, then free the GPU blocks.

        Args:
            gpu_block_ids: List of GPU block IDs to swap out

        Returns:
            Mapping of gpu_block_id → cpu_block_id

        Raises:
            ValueError: If not enough free CPU blocks

        TODO: Implement this.
        1. Allocate CPU blocks from block_manager.cpu_allocator
        2. For each (gpu_block, cpu_block) pair, for each layer:
              cpu_key_cache[layer][cpu_block] = gpu_key_cache[layer][gpu_block]
              cpu_value_cache[layer][cpu_block] = gpu_value_cache[layer][gpu_block]
        3. Free GPU blocks via block_manager.gpu_allocator
        4. Return mapping
        """
        raise NotImplementedError("TODO: Implement Swapper.swap_out")

    def swap_in(self, cpu_block_ids: List[int]) -> Dict[int, int]:
        """
        Copy blocks from CPU back to GPU, then free the CPU blocks.

        Args:
            cpu_block_ids: List of CPU block IDs to swap in

        Returns:
            Mapping of cpu_block_id → gpu_block_id

        TODO: Implement this.
        Mirror of swap_out.
        """
        raise NotImplementedError("TODO: Implement Swapper.swap_in")

    def swap_out_single(self, gpu_block_id: int) -> int:
        """Convenience: swap out a single block."""
        mapping = self.swap_out([gpu_block_id])
        return mapping[gpu_block_id]

    def swap_in_single(self, cpu_block_id: int) -> int:
        """Convenience: swap in a single block."""
        mapping = self.swap_in([cpu_block_id])
        return mapping[cpu_block_id]
