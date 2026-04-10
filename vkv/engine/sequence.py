"""
Phase 1, Part 5: Sequence — per-request state + block table.

Naming aligned with nano-vLLM/vLLM:
  nano-vLLM: Sequence (sequence.py) + SequenceStatus
  vLLM:      Sequence + SequenceGroup + SequenceStatus
  vkv:       Sequence + SequenceStatus (this file)

Key insight: In nano-vLLM, a Sequence holds BOTH the request metadata
(token_ids, status, sampling params) AND the block_table. This unifies
what was previously split between our BlockTable and the request object.
"""

from copy import copy
from enum import Enum, auto
from itertools import count
from typing import List, Optional, Tuple

from vkv.engine.block import num_blocks_for_tokens
from vkv.engine.block_manager import BlockManager
from vkv.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    Sequence lifecycle states.
    Directly matches nano-vLLM's SequenceStatus.

    State machine:
        WAITING → RUNNING → FINISHED
                ↓         ↑
              SWAPPED ────┘

    WAITING:  In queue, no GPU blocks allocated yet
    RUNNING:  Active on GPU, blocks allocated, generating tokens
    SWAPPED:  Preempted, KV cache moved to CPU (vkv extension, nano-vLLM doesn't have this)
    FINISHED: Generation complete, blocks deallocated
    """
    WAITING = auto()
    RUNNING = auto()
    SWAPPED = auto()
    FINISHED = auto()


class Sequence:
    """
    A single inference request and its KV cache block mapping.

    Combines nano-vLLM's Sequence (request state + block_table) with
    our block management operations (allocate, append, fork, free).

    nano-vLLM's Sequence fields (we keep the same ones):
        seq_id, status, token_ids, last_token, num_tokens,
        num_prompt_tokens, num_cached_tokens, block_table

    Our additions (for deeper KV cache management):
        fork() for COW prefix sharing
        explicit allocate/free methods

    Usage:
        >>> seq = Sequence(token_ids=[1, 2, 3, 4, 5], block_manager=mgr)
        >>> seq.allocate()                    # allocate blocks for prompt
        >>> block_id, slot = seq.append_token(new_token_id=6)  # decode step
        >>> seq.free()                        # release blocks

    Attributes:
        seq_id:             Auto-incremented unique ID (same as nano-vLLM)
        status:             SequenceStatus (same as nano-vLLM)
        token_ids:          Full token sequence (same as nano-vLLM)
        block_table:        List of physical block IDs (same as nano-vLLM)
        num_tokens:         Current total token count (same as nano-vLLM)
        num_prompt_tokens:  Original prompt length (same as nano-vLLM)
        num_cached_tokens:  Tokens served from prefix cache (same as nano-vLLM)
    """

    _counter = count()

    def __init__(
        self,
        token_ids: List[int],
        block_manager: BlockManager,
        sampling_params: Optional[SamplingParams] = None,
    ):
        """
        Initialize a Sequence.

        Mirrors nano-vLLM's Sequence.__init__:
            self.seq_id = next(Sequence.counter)
            self.status = SequenceStatus.WAITING
            self.token_ids = copy(token_ids)
            self.block_table = []
            ...

        Args:
            token_ids: Prompt token IDs
            block_manager: Reference to BlockManager for allocation
            sampling_params: Generation parameters

        TODO: Implement this.
        1. Assign seq_id from _counter
        2. Set status = WAITING
        3. Copy token_ids
        4. Store block_manager reference
        5. Initialize block_table = []
        6. Set num_tokens = len(token_ids)
        7. Set num_prompt_tokens = len(token_ids)
        8. Set num_cached_tokens = 0
        9. Store sampling_params (use default if None)
        """
        raise NotImplementedError("TODO: Implement Sequence.__init__")

    def __len__(self) -> int:
        """Return total token count. Same as nano-vLLM."""
        return self.num_tokens

    @property
    def is_finished(self) -> bool:
        """Same as nano-vLLM's is_finished property."""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        """Number of generated (non-prompt) tokens. Same as nano-vLLM."""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def last_token(self) -> int:
        """Last token in the sequence."""
        return self.token_ids[-1] if self.token_ids else -1

    @property
    def num_blocks(self) -> int:
        """
        Number of blocks needed for current token count.
        Same logic as nano-vLLM:
            (self.num_tokens + self.block_size - 1) // self.block_size
        """
        return num_blocks_for_tokens(self.num_tokens, self.block_manager.block_size)

    @property
    def last_block_num_tokens(self) -> int:
        """How many tokens are in the last block. Matches nano-vLLM."""
        if self.num_tokens == 0:
            return 0
        remainder = self.num_tokens % self.block_manager.block_size
        return remainder if remainder != 0 else self.block_manager.block_size

    # ----- Block management operations -----

    def allocate(self) -> None:
        """
        Allocate blocks for the prefill phase.

        Corresponds to the allocation logic inside nano-vLLM's
        BlockManager.allocate(seq), but from the Sequence side.

        TODO: Implement this.
        1. Compute how many blocks are needed: num_blocks_for_tokens(num_prompt_tokens, block_size)
        2. Allocate from block_manager
        3. Store block IDs in self.block_table
        4. Set status to RUNNING
        """
        raise NotImplementedError("TODO: Implement Sequence.allocate")

    def append_token(self, token_id: int) -> Tuple[int, int]:
        """
        Record that one new token has been generated.
        Allocate a new block if the current last block is full.

        Combines nano-vLLM's:
            Sequence.append_token(token_id)  → updates token_ids
            BlockManager.may_append(seq)     → allocates new block if needed

        Args:
            token_id: The newly generated token ID

        Returns:
            (block_id, slot_idx): Where to write the new token's KV data.

        TODO: Implement this.
        1. Compute slot_idx = num_tokens % block_size
        2. If slot_idx == 0 AND num_tokens > 0 → need new block
           (same logic as nano-vLLM: `len(seq) % block_size == 1` triggers allocation
            because nano-vLLM calls append_token BEFORE may_append — we combine them)
        3. Append token_id to self.token_ids
        4. self.num_tokens += 1
        5. Return (block_table[-1], slot_idx)
        """
        raise NotImplementedError("TODO: Implement Sequence.append_token")

    def fork(self) -> 'Sequence':
        """
        Create a fork for prefix sharing (Copy-on-Write).

        nano-vLLM doesn't have an explicit fork — it uses hash-based prefix sharing.
        vLLM has a similar concept via SequenceGroup.fork().

        The forked Sequence shares the same physical blocks,
        with ref counts incremented.

        Returns:
            A new Sequence sharing the same blocks.

        TODO: Implement this.
        1. Create new Sequence with same token_ids and block_manager
        2. Copy block_table (same physical block IDs)
        3. Copy num_tokens and num_cached_tokens
        4. For each block_id, call block_manager.inc_ref()
        5. Set new sequence status to RUNNING
        6. Return the new Sequence
        """
        raise NotImplementedError("TODO: Implement Sequence.fork")

    def free(self) -> None:
        """
        Release all blocks owned by this sequence.

        Corresponds to nano-vLLM's BlockManager.deallocate(seq):
            for block_id in reversed(seq.block_table):
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_block(block_id)
            seq.block_table.clear()

        TODO: Implement this.
        1. Call block_manager.free(self.block_table)
        2. Clear block_table
        3. Set num_tokens = 0
        4. Set status to FINISHED
        """
        raise NotImplementedError("TODO: Implement Sequence.free")
