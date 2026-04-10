"""
Phase 2 (stub): Scheduler — continuous batching request scheduler.

Naming aligned with nano-vLLM/vLLM:
  nano-vLLM: Scheduler (scheduler.py) — manages waiting/running queues
  vLLM:      Scheduler (scheduler.py) — with SchedulerOutputs, SchedulingBudget
  vkv:       Scheduler (this file)

nano-vLLM's scheduler core loop:
    def schedule(self) -> tuple[list[Sequence], bool]:
        # Try prefill first (from waiting queue)
        # If no prefill, do decode (from running queue)
        # Handle preemption if out of memory

This is a Phase 2 stub — implementation comes after Phase 1 is complete.
"""

from collections import deque
from typing import List, Tuple

from vkv.engine.sequence import Sequence, SequenceStatus
from vkv.engine.block_manager import BlockManager


class Scheduler:
    """
    Continuous batching scheduler.

    Manages two queues (matching nano-vLLM):
        waiting: deque[Sequence]  — sequences waiting for prefill
        running: deque[Sequence]  — sequences currently generating tokens

    Each schedule() call returns a batch to execute and whether it's prefill or decode.
    """

    def __init__(self, block_manager: BlockManager, max_num_seqs: int = 256):
        self.block_manager = block_manager
        self.max_num_seqs = max_num_seqs
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, seq: Sequence) -> None:
        """Add a new sequence to the waiting queue. Same as nano-vLLM."""
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        """Check if all sequences are done. Same as nano-vLLM."""
        return not self.waiting and not self.running

    def schedule(self) -> Tuple[List[Sequence], bool]:
        """
        Core scheduling logic. Returns (batch, is_prefill).

        Phase 2 TODO: implement continuous batching with:
        - Prefill scheduling (WAITING → RUNNING)
        - Decode scheduling (continue RUNNING sequences)
        - Preemption when out of GPU memory

        For now, raises NotImplementedError.
        """
        raise NotImplementedError("Phase 2: Implement Scheduler.schedule")
