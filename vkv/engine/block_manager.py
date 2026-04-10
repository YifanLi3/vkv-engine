"""
Phase 1, Part 4: BlockManager — the KV cache memory pool.

Naming aligned with nano-vLLM/vLLM:
  nano-vLLM: BlockManager (block_manager.py) — manages blocks + free list + hash
  vLLM:      BlockSpaceManager — manages block allocation for all sequences
  vkv:       BlockManager (this file) — pre-allocated pool + ref counting

Key difference from nano-vLLM:
  nano-vLLM's BlockManager doesn't own KV tensors (those live in model_runner).
  Our BlockManager owns the pre-allocated KV cache tensors, because we want
  to learn about GPU memory pool design. This is closer to vLLM's approach
  where the Worker holds cache tensors and BlockSpaceManager tracks metadata.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from vkv.config import ModelConfig, CacheConfig
from vkv.engine.block import num_blocks_for_tokens, kv_block_size_bytes
from vkv.engine.block_allocator import BlockAllocator


@dataclass
class PoolStats:
    """Statistics for the memory pool."""
    total_blocks: int = 0
    used_blocks: int = 0
    free_blocks: int = 0
    utilization: float = 0.0
    fragmentation: float = 0.0
    block_size_bytes: int = 0
    total_memory_bytes: int = 0
    used_memory_bytes: int = 0


class BlockManager:
    """
    Pre-allocated KV cache memory pool + block management.

    Corresponds to nano-vLLM's BlockManager, but with explicit KV tensor ownership.

    nano-vLLM structure:
        BlockManager.blocks: list[Block]           → metadata
        BlockManager.free_block_ids: deque[int]    → free list
        BlockManager.used_block_ids: set[int]      → in-use tracking

    Our structure:
        BlockManager.gpu_allocator: BlockAllocator → free list (separated for Part 3)
        BlockManager.gpu_key_cache: List[Tensor]   → actual KV data (pre-allocated)
        BlockManager.gpu_value_cache: List[Tensor]
        BlockManager._ref_counts: Dict[int, int]   → reference counting

    Storage layout (per layer):
        gpu_key_cache[layer_idx] shape: [num_gpu_blocks, num_kv_heads, block_size, head_dim]
        gpu_value_cache[layer_idx] shape: same

    Example:
        Block #5, Layer #3, Slot #7:
            key_data = self.gpu_key_cache[3][5, :, 7, :]
            # shape: [num_kv_heads, head_dim]
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        device: str = "cpu",
    ):
        """
        Pre-allocate the GPU (and CPU) memory pool.

        Args:
            model_config: Model architecture config (layers, heads, etc.)
            cache_config: Cache config (block_size, num_gpu_blocks, etc.)
            device: Device for the GPU pool ("cuda" or "cpu" for testing)

        TODO: You need to:
        1. Store configs and compute num_gpu_blocks
        2. Pre-allocate gpu_key_cache and gpu_value_cache:
           - List of tensors, one per layer
           - Each tensor shape: [num_gpu_blocks, num_kv_heads, block_size, head_dim]
        3. Similarly allocate cpu_key_cache and cpu_value_cache (for swap)
        4. Create BlockAllocators for GPU and CPU
        5. Create ref_count dict: Dict[int, int]

        In nano-vLLM, this pre-allocation happens in ModelRunner:
            self.key_caches = [torch.zeros(...) for _ in range(num_layers)]
            self.value_caches = [torch.zeros(...) for _ in range(num_layers)]
        We do the same, but inside BlockManager.
        """
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device
        self.block_size = cache_config.block_size
        self.num_layers = model_config.num_layers

        if cache_config.num_gpu_blocks is not None:
            self.num_gpu_blocks = cache_config.num_gpu_blocks
        else:
            raise ValueError("num_gpu_blocks must be specified when device is 'cpu'")

        self.num_cpu_blocks = cache_config.num_cpu_blocks

        # TODO: Allocate GPU KV cache tensors
        # self.gpu_key_cache: List[torch.Tensor] = [...]   # len = num_layers
        # self.gpu_value_cache: List[torch.Tensor] = [...]

        # TODO: Allocate CPU KV cache tensors (for swap)
        # self.cpu_key_cache: List[torch.Tensor] = [...]
        # self.cpu_value_cache: List[torch.Tensor] = [...]

        # TODO: Create BlockAllocators
        # self.gpu_allocator = BlockAllocator(self.num_gpu_blocks)
        # self.cpu_allocator = BlockAllocator(self.num_cpu_blocks)

        # TODO: Reference counting
        # self._ref_counts: Dict[int, int] = {}

        raise NotImplementedError("TODO: Implement BlockManager.__init__")

    # ----- allocation / deallocation (matches nano-vLLM allocate/deallocate) -----

    def allocate(self, num_blocks: int) -> List[int]:
        """
        Allocate GPU blocks and return their IDs.

        Corresponds to part of nano-vLLM's BlockManager.allocate(seq).
        The difference: nano-vLLM takes a Sequence and computes num_blocks internally;
        we take num_blocks directly (the Sequence layer handles the computation).

        TODO: Implement this.
        1. Call self.gpu_allocator.allocate(num_blocks)
        2. Initialize ref_count to 1 for each allocated block
        3. Return the block IDs
        """
        raise NotImplementedError("TODO: Implement BlockManager.allocate")

    def free(self, block_ids: List[int]) -> None:
        """
        Decrease ref count for given blocks. Free blocks whose count reaches 0.

        Corresponds to nano-vLLM's BlockManager.deallocate(seq), but operates
        on block_ids directly. nano-vLLM's version:
            for block_id in reversed(seq.block_table):
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_block(block_id)

        TODO: Implement this.
        For each block_id:
            ref_count -= 1
            if ref_count == 0 → actually free via gpu_allocator
        """
        raise NotImplementedError("TODO: Implement BlockManager.free")

    def inc_ref(self, block_id: int) -> None:
        """Increment reference count for a block (for prefix sharing).

        In nano-vLLM: block.ref_count += 1 (inside allocate, for cache hits)

        TODO: Implement this.
        """
        raise NotImplementedError("TODO: Implement BlockManager.inc_ref")

    def get_ref_count(self, block_id: int) -> int:
        """Get reference count for a block."""
        return self._ref_counts.get(block_id, 0)

    def can_allocate(self, num_blocks: int) -> bool:
        """Check if we can allocate num_blocks GPU blocks.

        Matches nano-vLLM's BlockManager.can_allocate(seq).
        """
        return self.gpu_allocator.has_free(num_blocks)

    # ----- KV data read/write -----

    def write_kv(
        self,
        block_id: int,
        layer_idx: int,
        slot_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Write one token's KV data into a specific block/layer/slot.

        Args:
            block_id:  Physical block ID in the pool
            layer_idx: Transformer layer index
            slot_idx:  Token slot within the block (0 to block_size-1)
            key:       Key tensor, shape [num_kv_heads, head_dim]
            value:     Value tensor, shape [num_kv_heads, head_dim]

        TODO: Implement this.
        Hint: self.gpu_key_cache[layer_idx][block_id, :, slot_idx, :] = key
        """
        raise NotImplementedError("TODO: Implement BlockManager.write_kv")

    def read_kv(
        self,
        block_id: int,
        layer_idx: int,
        slot_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read one token's KV data from a specific block/layer/slot.

        Returns:
            (key, value) tuple, each shape [num_kv_heads, head_dim]

        TODO: Implement this.
        """
        raise NotImplementedError("TODO: Implement BlockManager.read_kv")

    def gather_kv(
        self,
        block_ids: List[int],
        num_tokens: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather complete KV cache from a sequence of blocks for one layer.

        This is the operation that happens before standard attention computation.
        In nano-vLLM/vLLM, this gather is avoided by using PagedAttention kernels
        that read directly from the block table. But for correctness testing and
        Level 2 HuggingFace integration, we need this explicit gather.

        Args:
            block_ids:  Ordered list of block IDs for this sequence
            num_tokens: Total number of valid tokens across all blocks
            layer_idx:  Which transformer layer

        Returns:
            (keys, values) tuple:
                keys shape:   [1, num_kv_heads, num_tokens, head_dim]
                values shape: [1, num_kv_heads, num_tokens, head_dim]

        TODO: Implement this.
        For each block, slice the valid tokens and torch.cat them together.
        The last block may be partial (not all block_size slots filled).
        """
        raise NotImplementedError("TODO: Implement BlockManager.gather_kv")

    def copy_block(self, src_block_id: int, dst_block_id: int) -> None:
        """
        Copy all data from one block to another (for Copy-on-Write).

        TODO: Implement this.
        For each layer: copy key and value caches.
        """
        raise NotImplementedError("TODO: Implement BlockManager.copy_block")

    # ----- fragmentation (Bonus, Part 8) -----

    def compute_fragmentation(self, sequences) -> float:
        """
        Compute internal fragmentation ratio across all active sequences.

        Internal fragmentation = wasted slots / total allocated slots

        TODO: Implement this (Bonus).
        """
        raise NotImplementedError("TODO: Implement compute_fragmentation (Bonus)")

    # ----- stats -----

    @property
    def stats(self) -> PoolStats:
        """Get current pool statistics."""
        block_bytes = kv_block_size_bytes(self.model_config, self.block_size)
        used = self.gpu_allocator.num_used
        total = self.num_gpu_blocks
        return PoolStats(
            total_blocks=total,
            used_blocks=used,
            free_blocks=total - used,
            utilization=used / total if total > 0 else 0.0,
            block_size_bytes=block_bytes,
            total_memory_bytes=total * block_bytes,
            used_memory_bytes=used * block_bytes,
        )
