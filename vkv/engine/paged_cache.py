"""
Phase 6, Part 2: PagedCache — custom HuggingFace Cache implementation.

Replaces HuggingFace's DynamicCache (torch.cat based) with our
BlockManager-backed paged storage.

Injected via: model.generate(..., past_key_values=paged_cache)
"""

from typing import List, Optional, Tuple

import torch

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = object  # fallback for testing without transformers

from vkv.engine.block_manager import BlockManager


class PagedCache(Cache):
    """
    HuggingFace Cache backed by vkv-engine's BlockManager.

    DynamicCache:  torch.cat on every token → fragmentation
    PagedCache:    write to pre-allocated block slots → zero fragmentation

    The model's attention layers call cache.update() each layer each step.
    We redirect those writes to our BlockManager.
    """

    def __init__(
        self,
        block_manager: BlockManager,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
    ):
        """
        TODO: Implement this.
        1. Store block_manager and config
        2. Initialize block_table: List[int] = []
        3. Initialize _seq_length = 0
        """
        raise NotImplementedError("TODO: Implement PagedCache.__init__")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Write new KV data and return full KV cache for this layer.

        Called by HuggingFace model internals, once per layer per step.

        Args:
            key_states:   [batch=1, num_kv_heads, new_tokens, head_dim]
            value_states: same shape
            layer_idx:    which transformer layer

        Returns:
            (full_keys, full_values) for this layer, including history

        TODO: Implement this.
        1. For each new token:
           a. Compute block_idx and slot_idx
           b. Allocate new block if needed
           c. Write KV to block_manager
        2. Update _seq_length (only on last layer to avoid double-counting)
        3. Gather and return full KV using block_manager.gather_kv()
        """
        raise NotImplementedError("TODO: Implement PagedCache.update")

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return current cached sequence length.

        TODO: Implement this.
        """
        raise NotImplementedError("TODO: Implement get_seq_length")

    def get_max_cache_length(self) -> Optional[int]:
        """Max tokens this cache can hold."""
        return self.block_manager.num_gpu_blocks * self.block_size

    def free(self):
        """Release all blocks.

        TODO: Implement this.
        1. block_manager.free(self.block_table)
        2. Clear block_table
        3. Reset _seq_length
        """
        raise NotImplementedError("TODO: Implement PagedCache.free")
