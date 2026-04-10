"""
Phase 1, Part 6: Eviction Policies

When GPU memory is full, we need to evict some sequences' KV cache.

nano-vLLM uses simple preemption (deallocate + re-queue to waiting):
    def preempt(self, seq):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

vLLM has a more sophisticated eviction system with EvictionPolicy interface.

Our LRUEvictor is closer to vLLM's approach: it tracks access order and
selects victims based on recency, enabling smarter preemption decisions.
"""

from collections import OrderedDict
from typing import List, Tuple


class LRUEvictor:
    """
    Least Recently Used eviction policy.

    Tracks all evictable sequences (those whose KV cache is on GPU).
    When memory pressure occurs, evicts the least recently accessed sequences.

    Implementation: OrderedDict for O(1) access, move, and pop.

    Usage:
        >>> evictor = LRUEvictor()
        >>> evictor.add("seq_0", [0, 1, 2])
        >>> evictor.add("seq_1", [3, 4])
        >>> evictor.touch("seq_0")           # seq_0 is now most recent
        >>> victims = evictor.evict(3)        # need 3 blocks
        >>> # Returns [("seq_1", [3, 4])] — seq_1 is older
    """

    def __init__(self):
        """
        Initialize the LRU evictor.

        TODO: Implement this.
        Hint: self._entries = OrderedDict()
        """
        raise NotImplementedError("TODO: Implement LRUEvictor.__init__")

    def add(self, request_id: str, block_ids: List[int]) -> None:
        """
        Register a sequence as evictable.

        TODO: Implement this.
        """
        raise NotImplementedError("TODO: Implement LRUEvictor.add")

    def remove(self, request_id: str) -> None:
        """
        Remove a sequence from the evictor (e.g., when it finishes).
        Ignore silently if not found.

        TODO: Implement this.
        """
        raise NotImplementedError("TODO: Implement LRUEvictor.remove")

    def touch(self, request_id: str) -> None:
        """
        Mark a sequence as recently accessed (move to most-recent end).

        TODO: Implement this.
        Hint: self._entries.move_to_end(request_id)
        """
        raise NotImplementedError("TODO: Implement LRUEvictor.touch")

    def update_blocks(self, request_id: str, block_ids: List[int]) -> None:
        """
        Update the block list for a sequence (e.g., after appending new blocks).

        TODO: Implement this.
        """
        raise NotImplementedError("TODO: Implement LRUEvictor.update_blocks")

    def evict(self, num_blocks_needed: int) -> List[Tuple[str, List[int]]]:
        """
        Select sequences to evict to free at least `num_blocks_needed` blocks.

        Evicts from least-recently-used first. May need to evict multiple sequences.

        Returns:
            List of (request_id, block_ids) tuples for evicted sequences.

        Raises:
            ValueError: If all evictable sequences combined don't have enough blocks.

        TODO: Implement this.
        """
        raise NotImplementedError("TODO: Implement LRUEvictor.evict")

    @property
    def num_evictable_blocks(self) -> int:
        """Total number of blocks that could be freed by evicting everything.

        TODO: Implement this.
        """
        raise NotImplementedError("TODO: Implement num_evictable_blocks")

    def __len__(self) -> int:
        """Number of evictable sequences."""
        return len(self._entries)
