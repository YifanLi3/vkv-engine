"""
Phase 2: Scheduler — continuous batching request scheduler.

Naming aligned with nano-vLLM/vLLM:
  nano-vLLM: Scheduler (scheduler.py) — manages waiting/running queues
  vLLM:      Scheduler (scheduler.py) — with SchedulerOutputs, SchedulingBudget
  vkv:       Scheduler (this file)

nano-vLLM's scheduler has 3 methods:
  schedule()    → decide what to run this step
  preempt()     → evict a sequence when out of memory
  postprocess() → handle generated tokens, check EOS
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vkv.engine.block import num_blocks_for_tokens
from vkv.engine.sequence import Sequence, SequenceStatus
from vkv.engine.block_manager import BlockManager
from vkv.engine.evictor import LRUEvictor
from vkv.engine.swapper import Swapper


# =============================================================================
# Part 1: SchedulerConfig & SchedulerOutput
# =============================================================================

@dataclass
class SchedulerConfig:
    """
    Configuration for the scheduler.

    Attributes:
        max_num_seqs: Maximum sequences in one batch.
            nano-vLLM: config.max_num_seqs
        max_num_batched_tokens: Maximum tokens processed in one step.
            nano-vLLM: config.max_num_batched_tokens
        preemption_mode: How to handle preemption.
            "recompute" = nano-vLLM style (discard KV, re-prefill later)
            "swap"      = vkv extension (save KV to CPU)
        enable_chunked_prefill: Whether to enable chunked prefill (Part 6).
        chunk_size: Max tokens per prefill chunk when chunked prefill is enabled.
        eos_token_id: End-of-sequence token ID.

    """
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 4096
    preemption_mode: str = "recompute"
    enable_chunked_prefill: bool = False
    chunk_size: int = 512
    eos_token_id: int = 2


@dataclass
class SchedulerOutput:
    """
    Result of one schedule() call.

    nano-vLLM returns a simple (list, bool) tuple.
    We use a dataclass for clarity.

    Attributes:
        scheduled_seqs: Sequences to execute this step.
        is_prefill: True if this step is prefill, False if decode.
        preempted_seqs: Sequences that were preempted this step.
        swapped_in_seqs: Sequences swapped back from CPU this step.
        num_batched_tokens: Total tokens in this batch.

    """
    scheduled_seqs: List[Sequence] = field(default_factory=list)
    is_prefill: bool = False
    preempted_seqs: List[Sequence] = field(default_factory=list)
    swapped_in_seqs: List[Sequence] = field(default_factory=list)
    num_batched_tokens: int = 0


# =============================================================================
# Part 2-6: Scheduler
# =============================================================================

class Scheduler:
    """
    Continuous batching scheduler.

    Manages three queues (nano-vLLM has two, we add swapped):
        waiting: deque[Sequence]  — waiting for prefill
        running: deque[Sequence]  — actively generating tokens on GPU
        swapped: deque[Sequence]  — preempted, KV cache on CPU (vkv extension)

    Core loop (each step):
        1. Try swap_in (if swapped queue has sequences and GPU has space)
        2. Try prefill (if waiting queue has sequences)
        3. If no prefill, do decode (continue running sequences)
        4. If decode needs more blocks than available, preempt

    Matches nano-vLLM's Scheduler interface:
        add(seq)                     → add to waiting
        schedule()                   → returns (batch, is_prefill)
        postprocess(seqs, token_ids) → handle results
        is_finished()                → all done?
    """

    def __init__(
        self,
        block_manager: BlockManager,
        config: SchedulerConfig = None,
    ):
        """
        Initialize the scheduler.

        Args:
            block_manager: BlockManager from Phase 1
            config: Scheduler configuration

        1. Store block_manager and config (use default SchedulerConfig if None)
        2. Initialize three queues: waiting, running, swapped
        3. Create LRUEvictor instance
        4. Create Swapper instance (for swap mode)
        """
        self.block_manager = block_manager
        self.config = config

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.swapped: deque[Sequence] = deque()

        self.evictor = LRUEvictor()
        self.swapper = Swapper(block_manager)

    def add(self, seq: Sequence) -> None:
        """
        Add a new sequence to the waiting queue.
        Same as nano-vLLM's Scheduler.add().

        """
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        """
        Check if all sequences are done.
        Same as nano-vLLM: not self.waiting and not self.running

        Also check self.swapped queue (nano-vLLM doesn't have this).
        """
        if not self.waiting and not self.running and not self.swapped:
            return True
        else:
            return False

    # ---- Part 2: Prefill scheduling ----

    def _schedule_prefill(self) -> List[Sequence]:
        """
        Select sequences from waiting queue for prefill.

        This corresponds to the first half of nano-vLLM's schedule():
            while self.waiting:
                seq = self.waiting[0]
                if can_allocate(seq):
                    allocate(seq)
                    seq.status = RUNNING
                    self.waiting.popleft()
                    self.running.append(seq)

        Algorithm:
        1. Initialize num_seqs = 0, num_tokens = 0
        2. While waiting queue is not empty:
           a. Peek at the first waiting sequence
           b. Check constraints:
              - num_seqs + 1 <= max_num_seqs
              - num_tokens + len(seq) <= max_num_batched_tokens
              - block_manager can allocate enough blocks
           c. If all satisfied:
              - seq.allocate() — allocate blocks
              - Move from waiting to running
              - Add to scheduled list
              - Update evictor
           d. If any constraint fails: break
        3. Return scheduled list

        Returns:
            List of sequences to prefill this step.

        """
        num_seqs = 0
        num_tokens = 0
        scheduled = []

        while self.waiting:
            seq = self.waiting[0]
            if num_seqs + 1 > self.config.max_num_seqs:
                break
            if num_tokens + len(seq) > self.config.max_num_batched_tokens:
                break

            needed_blocks = num_blocks_for_tokens(len(seq), self.block_manager.block_size)
            if not self.block_manager.can_allocate(needed_blocks):
                break
            
            self.waiting.popleft()
            seq.allocate()
            self.running.append(seq)
            scheduled.append(seq)
            self.evictor.add(str(seq.seq_id), seq.block_table)

            num_seqs += 1
            num_tokens += len(seq)

        return scheduled

    # ---- Part 3: Decode scheduling + Preemption ----

    def _can_append(self, seq: Sequence) -> bool:
        """
        Check if a running sequence can append one more token.

        If the last block is full (num_tokens % block_size == 0),
        we need one free GPU block. Otherwise, no new block needed.

        Same logic as nano-vLLM's BlockManager.can_append(seq):
            return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

        Note: nano-vLLM checks `== 1` because it calls append_token BEFORE
        can_append. We check `== 0` because we check BEFORE appending.

        """
        if seq.num_tokens % self.block_manager.block_size == 0:
            return self.block_manager.gpu_allocator.has_free(1)
        else:
            return True

    def preempt(self, seq: Sequence) -> None:
        """
        Preempt a running sequence to free GPU memory.

        Two modes:
        - "recompute" (nano-vLLM style):
            seq.free()                    → deallocate all blocks
            seq.status = WAITING
            self.waiting.appendleft(seq)  → re-queue for later re-prefill

        - "swap" (vkv extension):
            mapping = swapper.swap_out(seq.block_table)
            seq.status = SWAPPED
            self.swapped.append(seq)

        TODO: Implement this.
        Check self.config.preemption_mode to decide which path.
        """
        raise NotImplementedError("TODO: Implement Scheduler.preempt")

    def _schedule_decode(self) -> Tuple[List[Sequence], List[Sequence]]:
        """
        Schedule decode step for running sequences.

        This corresponds to the second half of nano-vLLM's schedule():
            while self.running:
                seq = self.running.popleft()
                while not can_append(seq):
                    preempt(self.running.pop())
                scheduled.append(seq)

        Algorithm:
        1. Collect all running sequences
        2. For each, check if it can append a token (_can_append)
        3. If not → preempt the least recently used sequence (via evictor)
           until there's enough space
        4. Return (scheduled_seqs, preempted_seqs)

        Returns:
            (decode_seqs, preempted_seqs)

        TODO: Implement this.
        """
        raise NotImplementedError("TODO: Implement Scheduler._schedule_decode")

    def _try_swap_in(self) -> List[Sequence]:
        """
        Try to swap in sequences from the swapped queue.

        Only applies when preemption_mode == "swap".

        Algorithm:
        1. For each sequence in swapped queue:
           a. Calculate how many GPU blocks it needs
           b. If BlockManager has enough free blocks:
              - swapper.swap_in(seq.cpu_block_table) → get new GPU blocks
              - Update seq.block_table with new GPU block IDs
              - seq.status = RUNNING
              - Move from swapped to running
        2. Return list of swapped-in sequences

        TODO: Implement this.
        """
        raise NotImplementedError("TODO: Implement Scheduler._try_swap_in")

    def schedule(self) -> SchedulerOutput:
        """
        Core scheduling entry point. Called once per step.

        Matches nano-vLLM's schedule() signature pattern.

        Algorithm:
        1. Try swap_in (Part 3)
        2. Try prefill from waiting queue (Part 2)
        3. If no prefill, do decode from running queue (Part 3)

        Returns:
            SchedulerOutput with scheduled sequences and metadata.

        Wire together _try_swap_in, _schedule_prefill, _schedule_decode.
        """
        list_of_prefill_seq = self._schedule_prefill()
        if list_of_prefill_seq:
            return SchedulerOutput(
                scheduled_seqs=list_of_prefill_seq,
                is_prefill=True,
                preempted_seqs=[],
                swapped_in_seqs=[],
                num_batched_tokens=sum(s.num_tokens for s in list_of_prefill_seq)
            )


    # ---- Part 4: Postprocess ----

    def postprocess(
        self,
        scheduled_seqs: List[Sequence],
        token_ids: List[int],
    ) -> List[Sequence]:
        """
        Process generated tokens after a decode step.

        Directly matches nano-vLLM's Scheduler.postprocess():
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or \
                   seq.num_completion_tokens == seq.max_tokens:
                    seq.status = FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)

        Args:
            scheduled_seqs: Sequences that just ran a decode step
            token_ids: Generated token ID for each sequence

        Returns:
            List of sequences that finished this step.

        TODO: Implement this.
        1. For each (seq, token_id) pair:
           a. Call seq.append_token(token_id) — adds token + allocates block if needed
           b. Check if finished:
              - token_id == eos_token_id (and not ignore_eos)
              - seq.num_completion_tokens >= seq.sampling_params.max_tokens
           c. If finished:
              - seq.free() — release blocks
              - Remove from running queue
              - Remove from evictor
        2. Update evictor for sequences that are still running
        3. Return list of finished sequences
        """
        raise NotImplementedError("TODO: Implement Scheduler.postprocess")

    # ---- Part 6: Chunked Prefill (advanced) ----

    def _schedule_chunked_prefill(self) -> SchedulerOutput:
        """
        Mixed prefill + decode scheduling (Sarathi-style).

        Instead of doing prefill OR decode each step, do BOTH:
        1. Schedule decode first (each seq = 1 token)
        2. Use remaining token budget for a prefill chunk

        Algorithm:
        1. budget = max_num_batched_tokens
        2. Schedule decode: budget -= len(running_seqs)  (1 token each)
        3. Schedule prefill chunk:
           - Take first waiting seq
           - Prefill min(remaining_prompt, budget, chunk_size) tokens
           - If partially prefilled, keep in waiting with updated state
        4. Return combined batch

        TODO: Implement this (advanced).
        """
        raise NotImplementedError("TODO: Implement chunked prefill (advanced)")
