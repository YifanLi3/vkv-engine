"""
Phase 1, Part 3: BlockAllocator — free list manager for block IDs.

Naming aligned with vLLM:
  vLLM:      BlockAllocator (interface), NaiveBlockAllocator, PrefixCachingBlockAllocator
  nano-vLLM: inline in BlockManager (free_block_ids deque + used_block_ids set)
  vkv:       BlockAllocator (this file) — separated for pedagogical clarity

In nano-vLLM, the free list is just two fields in BlockManager:
    self.free_block_ids: deque[int] = deque(range(num_blocks))
    self.used_block_ids: set[int] = set()

We extract this into its own class so you can focus on the allocation logic first,
then compose it into BlockManager in Part 4.
"""

from typing import List
from unittest.signals import removeResult


class BlockAllocator:
    """
    O(1) block ID allocator using a free list.

    Tracks which block IDs (0 to num_blocks-1) are free vs in-use.
    Does NOT touch actual tensor memory — that's BlockManager's job.

    Corresponds to nano-vLLM's free_block_ids/used_block_ids inside BlockManager.

    Usage:
        >>> alloc = BlockAllocator(num_blocks=100)
        >>> ids = alloc.allocate(3)   # e.g. [0, 1, 2]
        >>> alloc.num_free
        97
        >>> alloc.free(ids)
        >>> alloc.num_free
        100
    """

    def __init__(self, num_blocks: int):
        """
        Initialize the allocator with all block IDs available.

        Args:
            num_blocks: Total number of blocks to manage (IDs: 0 to num_blocks-1)

        TODO: You need to:
        1. Store num_blocks
        2. Initialize a data structure containing all block IDs as "free"
        3. Optionally maintain a set of in-use IDs for O(1) double-free detection

        nano-vLLM uses:
          self.free_block_ids: deque[int] = deque(range(num_blocks))
          self.used_block_ids: set[int] = set()

        You can use the same approach, or use a list-as-stack (LIFO).
        LIFO is slightly better for GPU cache locality: recently freed blocks
        get reused first, meaning GPU cache lines may still be warm.
        """
        self._num_blocks = num_blocks
        self._free_block_ids = list(range(num_blocks))
        self._used_block_ids = set()

    def allocate(self, num_blocks: int) -> List[int]:
        """
        Allocate `num_blocks` block IDs from the free list.

        Args:
            num_blocks: How many blocks to allocate

        Returns:
            List of allocated block IDs

        Raises:
            ValueError: If not enough free blocks available

        TODO: Implement this.
        1. Check that num_blocks <= self.num_free, else raise ValueError
        2. Pop num_blocks IDs from your free list
        3. Add them to the in-use tracking set
        4. Return the list of IDs
        """
        if num_blocks > self.num_free:
            raise ValueError("No enough free blocks")

        result = []
        for _ in range(num_blocks):
            block_id = self._free_block_ids.pop()
            self._used_block_ids.add(block_id)
            result.append(block_id)

        return result

    def free(self, block_ids: List[int]) -> None:
        """
        Return block IDs to the free list.

        Args:
            block_ids: List of block IDs to free

        Raises:
            ValueError: If any block ID is not currently allocated (double free)

        1. For each block_id, check it's in the in-use set, else raise ValueError
        2. Remove from in-use set
        3. Push back onto the free list
        """
        for ids in block_ids:
            if ids not in self._used_block_ids:
                raise ValueError("This block id is already freed")
            self._used_block_ids.remove(ids)
            self._free_block_ids.append(ids)

    @property
    def num_free(self) -> int:
        """Number of currently free blocks.
        """
        return len(self._free_block_ids)

    @property
    def num_used(self) -> int:
        """Number of currently allocated blocks.
        """
        return len(self._used_block_ids)

    @property
    def total(self) -> int:
        """Total number of blocks managed."""
        return self._num_blocks

    def has_free(self, num_blocks: int = 1) -> bool:
        """Check if at least `num_blocks` free blocks are available."""
        return self.num_free >= num_blocks
