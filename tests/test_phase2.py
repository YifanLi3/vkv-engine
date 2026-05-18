"""
Phase 2 Test Suite — Continuous Batching Scheduler

Run all tests:       uv run pytest tests/test_phase2.py -v
Run one Part:        uv run pytest tests/test_phase2.py -k "part1" -v
"""

import pytest
from collections import deque

from vkv.config import ModelConfig, CacheConfig, TINY_MODEL
from vkv.sampling_params import SamplingParams
from vkv.engine.block_manager import BlockManager
from vkv.engine.sequence import Sequence, SequenceStatus
from vkv.engine.scheduler import Scheduler, SchedulerConfig, SchedulerOutput


def make_block_manager(num_gpu_blocks=100):
    return BlockManager(
        TINY_MODEL,
        CacheConfig(block_size=16, num_gpu_blocks=num_gpu_blocks, num_cpu_blocks=50),
        device="cpu",
    )


def make_scheduler(num_gpu_blocks=100, **kwargs):
    mgr = make_block_manager(num_gpu_blocks)
    config = SchedulerConfig(**kwargs)
    return Scheduler(mgr, config), mgr


def make_seq(mgr, prompt_len=32, max_tokens=64):
    token_ids = list(range(prompt_len))
    sp = SamplingParams(max_tokens=max_tokens)
    return Sequence(token_ids=token_ids, block_manager=mgr, sampling_params=sp)


# =============================================================================
# Part 1: SchedulerConfig & SchedulerOutput
# =============================================================================

class TestPart1:
    """Tests for Part 1: Config and Output data structures."""

    def test_scheduler_config_defaults(self):
        config = SchedulerConfig()
        assert config.max_num_seqs == 256
        assert config.max_num_batched_tokens == 4096
        assert config.preemption_mode == "recompute"

    def test_scheduler_config_custom(self):
        config = SchedulerConfig(max_num_seqs=32, max_num_batched_tokens=2048)
        assert config.max_num_seqs == 32
        assert config.max_num_batched_tokens == 2048

    def test_scheduler_output_defaults(self):
        output = SchedulerOutput()
        assert output.scheduled_seqs == []
        assert output.is_prefill is False
        assert output.preempted_seqs == []

    def test_scheduler_init(self):
        sched, mgr = make_scheduler()
        assert len(sched.waiting) == 0
        assert len(sched.running) == 0
        assert sched.is_finished()

    def test_add_sequence(self):
        sched, mgr = make_scheduler()
        seq = make_seq(mgr)
        sched.add(seq)
        assert len(sched.waiting) == 1
        assert not sched.is_finished()


# =============================================================================
# Part 2: Prefill Scheduling
# =============================================================================

class TestPart2:
    """Tests for Part 2: _schedule_prefill and schedule (prefill path)."""

    def test_prefill_single_sequence(self):
        sched, mgr = make_scheduler()
        seq = make_seq(mgr, prompt_len=32)
        sched.add(seq)

        output = sched.schedule()
        assert output.is_prefill is True
        assert len(output.scheduled_seqs) == 1
        assert output.scheduled_seqs[0] is seq
        assert seq.status == SequenceStatus.RUNNING
        assert len(seq.block_table) > 0

    def test_prefill_multiple_sequences(self):
        sched, mgr = make_scheduler(max_num_seqs=4)
        seqs = [make_seq(mgr, prompt_len=16) for _ in range(3)]
        for s in seqs:
            sched.add(s)

        output = sched.schedule()
        assert output.is_prefill is True
        assert len(output.scheduled_seqs) == 3
        assert len(sched.waiting) == 0
        assert len(sched.running) == 3

    def test_prefill_respects_max_num_seqs(self):
        sched, mgr = make_scheduler(max_num_seqs=2)
        seqs = [make_seq(mgr, prompt_len=16) for _ in range(5)]
        for s in seqs:
            sched.add(s)

        output = sched.schedule()
        assert len(output.scheduled_seqs) == 2
        assert len(sched.waiting) == 3

    def test_prefill_respects_max_batched_tokens(self):
        sched, mgr = make_scheduler(max_num_batched_tokens=50)
        seq_short = make_seq(mgr, prompt_len=30)
        seq_long = make_seq(mgr, prompt_len=40)
        sched.add(seq_short)
        sched.add(seq_long)

        output = sched.schedule()
        assert len(output.scheduled_seqs) == 1
        assert output.scheduled_seqs[0] is seq_short

    def test_prefill_respects_block_availability(self):
        sched, mgr = make_scheduler(num_gpu_blocks=3)
        seq = make_seq(mgr, prompt_len=64)  # needs ceil(64/16) = 4 blocks
        sched.add(seq)

        output = sched.schedule()
        assert len(output.scheduled_seqs) == 0
        assert len(sched.waiting) == 1

    def test_prefill_moves_to_running(self):
        sched, mgr = make_scheduler()
        seq = make_seq(mgr, prompt_len=32)
        sched.add(seq)

        sched.schedule()
        assert seq.status == SequenceStatus.RUNNING
        assert len(sched.running) == 1
        assert len(sched.waiting) == 0


# =============================================================================
# Part 3: Decode Scheduling + Preemption
# =============================================================================

class TestPart3:
    """Tests for Part 3: _schedule_decode, preempt, _can_append."""

    def _prefill_seq(self, sched, mgr, prompt_len=16):
        """Helper: add and prefill a sequence."""
        seq = make_seq(mgr, prompt_len=prompt_len, max_tokens=128)
        sched.add(seq)
        sched.schedule()  # prefill
        return seq

    def test_decode_after_prefill(self):
        sched, mgr = make_scheduler()
        seq = self._prefill_seq(sched, mgr, prompt_len=16)

        output = sched.schedule()
        assert output.is_prefill is False
        assert len(output.scheduled_seqs) == 1
        assert output.scheduled_seqs[0] is seq

    def test_decode_multiple_sequences(self):
        sched, mgr = make_scheduler(max_num_seqs=4)
        seqs = []
        for _ in range(3):
            seqs.append(self._prefill_seq(sched, mgr, prompt_len=16))

        output = sched.schedule()
        assert output.is_prefill is False
        assert len(output.scheduled_seqs) == 3

    def test_can_append_within_block(self):
        """If last block has room, no new block needed."""
        sched, mgr = make_scheduler()
        seq = self._prefill_seq(sched, mgr, prompt_len=10)
        assert sched._can_append(seq) is True

    def test_can_append_needs_new_block(self):
        """If last block is full, need a free block."""
        sched, mgr = make_scheduler(num_gpu_blocks=2)
        seq = self._prefill_seq(sched, mgr, prompt_len=16)
        # seq uses 1 block (16/16 = full), needs 1 more for next token
        # 2 total - 1 used = 1 free → can append
        assert sched._can_append(seq) is True

    def test_preempt_recompute_mode(self):
        sched, mgr = make_scheduler(preemption_mode="recompute")
        seq = self._prefill_seq(sched, mgr, prompt_len=16)
        blocks_before = mgr.stats.used_blocks

        sched.preempt(seq)

        assert seq.status == SequenceStatus.WAITING
        assert mgr.stats.used_blocks < blocks_before
        assert seq in sched.waiting

    def test_preempt_frees_memory(self):
        """Preempting a sequence should free its blocks."""
        sched, mgr = make_scheduler(num_gpu_blocks=10, preemption_mode="recompute")
        seq_a = self._prefill_seq(sched, mgr, prompt_len=32)  # 2 blocks
        seq_b = self._prefill_seq(sched, mgr, prompt_len=32)  # 2 blocks
        free_before = mgr.stats.free_blocks

        sched.preempt(seq_b)

        assert mgr.stats.free_blocks > free_before

    def test_decode_triggers_preemption(self):
        """When no free blocks for decode, should preempt a sequence."""
        sched, mgr = make_scheduler(num_gpu_blocks=4, preemption_mode="recompute")
        seq_a = self._prefill_seq(sched, mgr, prompt_len=16)  # 1 block
        seq_b = self._prefill_seq(sched, mgr, prompt_len=16)  # 1 block
        seq_c = self._prefill_seq(sched, mgr, prompt_len=16)  # 1 block
        seq_d = self._prefill_seq(sched, mgr, prompt_len=16)  # 1 block
        # All 4 blocks used, no free blocks

        # Each seq has 16 tokens (block full), next append needs a new block
        # Should trigger preemption
        output = sched.schedule()
        assert len(output.preempted_seqs) > 0 or len(output.scheduled_seqs) < 4


# =============================================================================
# Part 4: Postprocess
# =============================================================================

class TestPart4:
    """Tests for Part 4: postprocess."""

    def _setup_decode(self, sched, mgr, prompt_len=10, max_tokens=5):
        seq = make_seq(mgr, prompt_len=prompt_len, max_tokens=max_tokens)
        sched.add(seq)
        sched.schedule()  # prefill
        return seq

    def test_postprocess_appends_token(self):
        sched, mgr = make_scheduler()
        seq = self._setup_decode(sched, mgr, prompt_len=10)
        initial_tokens = seq.num_tokens

        sched.postprocess([seq], [42])

        assert seq.num_tokens == initial_tokens + 1
        assert seq.token_ids[-1] == 42

    def test_postprocess_detects_eos(self):
        sched, mgr = make_scheduler(eos_token_id=2)
        seq = self._setup_decode(sched, mgr, prompt_len=10)

        finished = sched.postprocess([seq], [2])  # EOS token

        assert len(finished) == 1
        assert seq.status == SequenceStatus.FINISHED
        assert seq not in sched.running

    def test_postprocess_detects_max_tokens(self):
        sched, mgr = make_scheduler()
        seq = self._setup_decode(sched, mgr, prompt_len=10, max_tokens=3)

        sched.postprocess([seq], [10])  # token 1
        sched.postprocess([seq], [20])  # token 2
        finished = sched.postprocess([seq], [30])  # token 3 → max_tokens reached

        assert len(finished) == 1
        assert seq.status == SequenceStatus.FINISHED

    def test_postprocess_not_finished(self):
        sched, mgr = make_scheduler()
        seq = self._setup_decode(sched, mgr, prompt_len=10, max_tokens=100)

        finished = sched.postprocess([seq], [42])

        assert len(finished) == 0
        assert seq.status == SequenceStatus.RUNNING

    def test_postprocess_frees_blocks_on_finish(self):
        sched, mgr = make_scheduler(eos_token_id=2)
        seq = self._setup_decode(sched, mgr, prompt_len=10)
        free_before = mgr.stats.free_blocks

        sched.postprocess([seq], [2])  # EOS

        assert mgr.stats.free_blocks > free_before

    def test_postprocess_multiple_seqs(self):
        sched, mgr = make_scheduler(eos_token_id=2)
        seq_a = self._setup_decode(sched, mgr, prompt_len=10)
        seq_b = self._setup_decode(sched, mgr, prompt_len=10)

        finished = sched.postprocess([seq_a, seq_b], [42, 2])

        assert len(finished) == 1  # only seq_b finished (EOS)
        assert seq_a.status == SequenceStatus.RUNNING
        assert seq_b.status == SequenceStatus.FINISHED


# =============================================================================
# Part 5: LLMEngine
# =============================================================================

class TestPart5:
    """Tests for Part 5: LLMEngine."""

    def test_engine_init(self):
        from vkv.engine.llm_engine import LLMEngine
        engine = LLMEngine(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=100))
        assert engine.is_finished()

    def test_engine_add_request(self):
        from vkv.engine.llm_engine import LLMEngine
        engine = LLMEngine(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=100))
        seq_id = engine.add_request([1, 2, 3, 4])
        assert isinstance(seq_id, int)
        assert not engine.is_finished()

    def test_engine_step_prefill(self):
        from vkv.engine.llm_engine import LLMEngine
        engine = LLMEngine(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=100))
        engine.add_request([1, 2, 3, 4])
        outputs = engine.step()  # should do prefill
        assert isinstance(outputs, list)

    def test_engine_step_decode(self):
        from vkv.engine.llm_engine import LLMEngine
        engine = LLMEngine(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=100))
        engine.add_request([1, 2, 3, 4], SamplingParams(max_tokens=3))
        engine.step()   # prefill
        engine.step()   # decode step 1

    def test_engine_generate_single(self):
        from vkv.engine.llm_engine import LLMEngine
        engine = LLMEngine(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=100))
        outputs = engine.generate(
            prompts=[[1, 2, 3, 4]],
            sampling_params=SamplingParams(max_tokens=5),
        )
        assert len(outputs) == 1
        assert len(outputs[0].output_token_ids) == 5

    def test_engine_generate_multiple(self):
        from vkv.engine.llm_engine import LLMEngine
        engine = LLMEngine(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=100))
        outputs = engine.generate(
            prompts=[[1, 2, 3], [4, 5, 6, 7, 8]],
            sampling_params=SamplingParams(max_tokens=3),
        )
        assert len(outputs) == 2
        for out in outputs:
            assert len(out.output_token_ids) == 3

    def test_engine_no_memory_leak(self):
        from vkv.engine.llm_engine import LLMEngine
        engine = LLMEngine(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=100))
        engine.generate(
            prompts=[[1, 2, 3], [4, 5, 6]],
            sampling_params=SamplingParams(max_tokens=5),
        )
        assert engine.scheduler.block_manager.stats.used_blocks == 0


# =============================================================================
# Part 6: Chunked Prefill (Advanced)
# =============================================================================

class TestPart6:
    """Tests for Part 6: Chunked Prefill."""

    def test_chunked_prefill_basic(self):
        sched, mgr = make_scheduler(
            enable_chunked_prefill=True,
            chunk_size=16,
            max_num_batched_tokens=32,
        )
        seq = make_seq(mgr, prompt_len=64)  # 4 chunks of 16
        sched.add(seq)

        output = sched.schedule()
        assert output.is_prefill is True
        assert output.num_batched_tokens <= 32


# =============================================================================
# Part 7: End-to-End Integration
# =============================================================================

class TestPart7:
    """Tests for Part 7: End-to-end simulation."""

    def test_e2e_10_requests(self):
        """Run 10 requests to completion, verify no leaks."""
        from vkv.engine.llm_engine import LLMEngine
        engine = LLMEngine(
            TINY_MODEL,
            CacheConfig(block_size=16, num_gpu_blocks=50),
            SchedulerConfig(max_num_seqs=4),
        )
        prompts = [list(range(i * 5, i * 5 + 10)) for i in range(10)]
        outputs = engine.generate(
            prompts=prompts,
            sampling_params=SamplingParams(max_tokens=10),
        )
        assert len(outputs) == 10
        for out in outputs:
            assert len(out.output_token_ids) == 10
        assert engine.scheduler.block_manager.stats.used_blocks == 0

    def test_e2e_mixed_lengths(self):
        """Requests with varying prompt lengths."""
        from vkv.engine.llm_engine import LLMEngine
        engine = LLMEngine(
            TINY_MODEL,
            CacheConfig(block_size=16, num_gpu_blocks=100),
            SchedulerConfig(max_num_seqs=8),
        )
        prompts = [
            list(range(5)),     # short
            list(range(50)),    # medium
            list(range(100)),   # long
            list(range(10)),    # short
        ]
        outputs = engine.generate(
            prompts=prompts,
            sampling_params=SamplingParams(max_tokens=5),
        )
        assert len(outputs) == 4
        assert engine.scheduler.block_manager.stats.used_blocks == 0

    def test_e2e_memory_pressure(self):
        """Many requests competing for limited GPU blocks."""
        from vkv.engine.llm_engine import LLMEngine
        engine = LLMEngine(
            TINY_MODEL,
            CacheConfig(block_size=16, num_gpu_blocks=10),
            SchedulerConfig(max_num_seqs=4, preemption_mode="recompute"),
        )
        prompts = [list(range(20)) for _ in range(8)]
        outputs = engine.generate(
            prompts=prompts,
            sampling_params=SamplingParams(max_tokens=5),
        )
        assert len(outputs) == 8
        assert engine.scheduler.block_manager.stats.used_blocks == 0
