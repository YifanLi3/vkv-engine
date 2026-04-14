"""
Phase 1, Part 1 & 2: Block — the basic unit of KV cache storage.

Naming aligned with nano-vLLM/vLLM:
  nano-vLLM: Block (in block_manager.py)
  vLLM:      PhysicalTokenBlock
  vkv:       Block (this file)
"""

import math
from typing import Tuple

import torch

from vkv.config import ModelConfig


# =============================================================================
# Part 1: KV Cache Size Calculations
# =============================================================================

def kv_cache_size_per_token(config: ModelConfig) -> int:
    """
    Compute the KV cache memory (in bytes) for a SINGLE token across ALL layers.

    A single token's KV cache at one layer consists of:
      - Key:   [num_kv_heads, head_dim]  (one tensor)
      - Value: [num_kv_heads, head_dim]  (one tensor)

    The total is summed across all layers.

    Args:
        config: Model configuration with num_layers, num_kv_heads, head_dim, dtype

    Returns:
        Total bytes for one token's KV cache across all layers.

    Example:
        >>> from vkv.config import LLAMA_8B
        >>> kv_cache_size_per_token(LLAMA_8B)
        131072 bytes => 128 KB

    Explanation for Llama-8B:
        32 layers x 2 (K+V) x 8 heads x 128 dim x 2 bytes(FP16) = 131072
    """

    # Hint: config.element_size gives you bytes per element for the configured dtype.
    return config.num_layers * 2 * config.num_kv_heads * config.head_dim * config.element_size


def num_blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    """
    Compute how many blocks are needed to store `num_tokens` tokens.
    Must round up (ceiling division).

    Args:
        num_tokens: Total number of tokens to store
        block_size: Number of tokens per block

    Returns:
        Number of blocks needed (ceiling division)

    Example:
        >>> num_blocks_for_tokens(50, 16)
        4  # ceil(50/16) = 4
        >>> num_blocks_for_tokens(32, 16)
        2  # exactly 2 blocks, no waste
        >>> num_blocks_for_tokens(0, 16)
        0
    """

    return (num_tokens + block_size - 1) // block_size


def kv_block_size_bytes(config: ModelConfig, block_size: int) -> int:
    """
    Compute the memory (in bytes) for a SINGLE KV block.

    One block stores `block_size` tokens worth of KV across all layers.

    Args:
        config: Model configuration
        block_size: Number of tokens per block

    Returns:
        Bytes for one complete block.

    Example:
        >>> from vkv.config import LLAMA_8B
        >>> kv_block_size_bytes(LLAMA_8B, 16)
        2097152  # 2 MB
    """

    # Hint: This is just kv_cache_size_per_token * block_size.
    return kv_cache_size_per_token(config) * block_size


# =============================================================================
# Part 2: Block Data Structure
# =============================================================================

class Block:
    """
    A single KV cache block that can hold `block_size` tokens.

    Naming: matches nano-vLLM's `Block` class in block_manager.py.

    nano-vLLM's Block is minimal (just block_id, ref_count, hash, token_ids)
    because its KV tensors live in a pre-allocated pool managed by ModelRunner.

    Our Block in Part 2 holds its own tensors for learning purposes.
    In Part 4 (BlockManager), we switch to the pooled approach — just like
    nano-vLLM and vLLM do in production.

    Storage layout:
        key_cache:   [num_layers, num_kv_heads, block_size, head_dim]
        value_cache: [num_layers, num_kv_heads, block_size, head_dim]

    Attributes:
        block_id:     Unique identifier (same as nano-vLLM)
        block_size:   Max tokens this block can hold
        num_filled:   How many token slots are currently written
        ref_count:    Reference count for prefix sharing (same as nano-vLLM)
        last_access:  Timestamp of last access (for LRU eviction)
    """

    def __init__(
        self,
        block_id: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ):
        """
        Initialize a Block and allocate its tensor storage.

        TODO: You need to:
        1. Store all the config parameters as instance attributes
        2. Allocate self.key_cache as a zero tensor with shape:
           [num_layers, num_kv_heads, block_size, head_dim]
        3. Allocate self.value_cache with the same shape
        4. Initialize num_filled = 0, ref_count = 1, last_access = 0.0
        """
        self.block_id = block_id
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Shape: [num_layers, num_kv_heads, block_size, head_dim]
        self.key_cache: torch.Tensor = torch.zeros(
                            num_layers, num_kv_heads, block_size, head_dim,
                            dtype=dtype, device=device,
                        )
        self.value_cache: torch.Tensor = torch.zeros(
                            num_layers, num_kv_heads, block_size, head_dim,
                            dtype=dtype, device=device,
                        )

        self.num_filled: int = 0
        self.ref_count: int = 1
        self.last_access: float = 0.0


    @property
    def is_full(self) -> bool:
        """Check if this block has no more free slots."""
        return self.num_filled >= self.block_size

    @property
    def num_empty_slots(self) -> int:
        """Number of unfilled token slots."""
        return self.block_size - self.num_filled

    def write_slot(
        self,
        layer_idx: int,
        slot_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Write one token's KV data into a specific slot at a specific layer.

        Args:
            layer_idx: Which transformer layer (0 to num_layers-1)
            slot_idx:  Which slot within this block (0 to block_size-1)
            key:       Key tensor, shape [num_kv_heads, head_dim]
            value:     Value tensor, shape [num_kv_heads, head_dim]

        Example:
            >>> block = Block(block_id=0, num_layers=2, num_kv_heads=4,
            ...               head_dim=64, block_size=16)
            >>> k = torch.randn(4, 64)
            >>> v = torch.randn(4, 64)
            >>> block.write_slot(layer_idx=0, slot_idx=0, key=k, value=v)
            >>> assert torch.equal(block.key_cache[0, :, 0, :], k)

        TODO: Implement this method.
        Hint: self.key_cache[layer_idx, :, slot_idx, :] = key
        Don't forget to validate that slot_idx < block_size.
        """
        raise NotImplementedError("TODO: Implement Block.write_slot")

    def read_slot(
        self,
        layer_idx: int,
        slot_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read one token's KV data from a specific slot.

        Returns:
            (key, value) tuple, each of shape [num_kv_heads, head_dim]

        TODO: Implement this method.
        """
        raise NotImplementedError("TODO: Implement Block.read_slot")

    def clear(self) -> None:
        """
        Clear this block, making it reusable.
        Analogous to nano-vLLM's Block.reset().

        TODO: Reset num_filled to 0.
        """
        raise NotImplementedError("TODO: Implement Block.clear")

    def memory_bytes(self) -> int:
        """Total memory used by this block's tensors (both key and value)."""
        return self.key_cache.nelement() * self.key_cache.element_size() * 2
