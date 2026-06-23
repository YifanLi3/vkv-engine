"""
Phase 5 Test Suite — Monitoring & Analysis

Run all tests:       uv run pytest tests/test_phase5.py -v
Run one Part:        uv run pytest tests/test_phase5.py -k "part1" -v
"""

import time
import pytest

from vkv.config import CacheConfig, TINY_MODEL
from vkv.sampling_params import SamplingParams


# =============================================================================
# Part 1: MetricsCollector
# =============================================================================

class TestPart1:
    """Tests for MetricsCollector."""

    def test_init(self):
        from vkv.engine.monitor import MetricsCollector
        collector = MetricsCollector()
        assert len(collector._request_metrics) == 0
        assert len(collector._completed_metrics) == 0

    def test_on_request_arrival(self):
        from vkv.engine.monitor import MetricsCollector
        collector = MetricsCollector()
        collector.on_request_arrival(seq_id=0, num_prompt_tokens=50)
        m = collector.get_request_metrics(0)
        assert m is not None
        assert m.seq_id == 0
        assert m.num_prompt_tokens == 50
        assert m.arrival_time > 0

    def test_on_first_token(self):
        from vkv.engine.monitor import MetricsCollector
        collector = MetricsCollector()
        collector.on_request_arrival(seq_id=0, num_prompt_tokens=50)
        time.sleep(0.01)
        collector.on_first_token(seq_id=0)
        m = collector.get_request_metrics(0)
        assert m.first_token_time > m.arrival_time
        assert m.ttft > 0

    def test_on_request_finish(self):
        from vkv.engine.monitor import MetricsCollector
        collector = MetricsCollector()
        collector.on_request_arrival(seq_id=0, num_prompt_tokens=50)
        collector.on_first_token(seq_id=0)
        time.sleep(0.01)
        collector.on_request_finish(seq_id=0, num_output_tokens=30)
        m = collector.get_request_metrics(0)
        assert m.finish_time > m.first_token_time
        assert m.num_output_tokens == 30
        assert m.tpot > 0
        assert collector._num_completed == 1

    def test_on_preemption(self):
        from vkv.engine.monitor import MetricsCollector
        collector = MetricsCollector()
        collector.on_preemption()
        collector.on_preemption()
        assert collector._num_preemptions == 2

    def test_collect_snapshot(self):
        from vkv.engine.monitor import MetricsCollector
        from vkv.engine.block_manager import BlockManager
        from vkv.engine.scheduler import Scheduler, SchedulerConfig
        from vkv.engine.sequence import Sequence

        mgr = BlockManager(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=100, num_cpu_blocks=50), device="cpu")
        sched = Scheduler(mgr, SchedulerConfig())

        seq = Sequence(token_ids=list(range(10)), block_manager=mgr)
        sched.add(seq)

        collector = MetricsCollector()
        snapshot = collector.collect_snapshot(mgr, sched)

        assert snapshot.total_blocks == 100
        assert snapshot.num_waiting == 1
        assert snapshot.num_running == 0
        assert snapshot.timestamp > 0

    def test_multiple_requests(self):
        from vkv.engine.monitor import MetricsCollector
        collector = MetricsCollector()

        for i in range(5):
            collector.on_request_arrival(seq_id=i, num_prompt_tokens=10 + i)
        assert len(collector._request_metrics) == 5

        for i in range(5):
            collector.on_first_token(seq_id=i)
            collector.on_request_finish(seq_id=i, num_output_tokens=5)
        assert collector._num_completed == 5
        assert len(collector._completed_metrics) == 5

    def test_get_summary(self):
        from vkv.engine.monitor import MetricsCollector
        collector = MetricsCollector()
        for i in range(3):
            collector.on_request_arrival(seq_id=i, num_prompt_tokens=10)
            time.sleep(0.01)
            collector.on_first_token(seq_id=i)
            time.sleep(0.01)
            collector.on_request_finish(seq_id=i, num_output_tokens=5)

        summary = collector.get_summary()
        assert "avg_ttft" in summary
        assert "avg_tpot" in summary
        assert summary["num_completed"] == 3
        assert summary["avg_ttft"] > 0
        assert summary["avg_tpot"] > 0


# =============================================================================
# Part 2: Prometheus Exporter
# =============================================================================

class TestPart2:
    """Tests for PrometheusExporter."""

    def test_update_gauges(self):
        from vkv.engine.monitor import PrometheusExporter, SystemSnapshot, RequestMetrics
        exporter = PrometheusExporter(port=0)

        snapshot = SystemSnapshot(
            timestamp=time.time(),
            total_blocks=100,
            used_blocks=60,
            free_blocks=40,
            utilization=0.6,
            num_waiting=5,
            num_running=10,
            num_swapped=2,
        )

        exporter.update(snapshot)

        assert exporter.block_utilization._value.get() == 0.6
        assert exporter.num_waiting._value.get() == 5
        assert exporter.num_running._value.get() == 10

    def test_update_with_completed_requests(self):
        from vkv.engine.monitor import PrometheusExporter, SystemSnapshot, RequestMetrics
        exporter = PrometheusExporter(port=0)

        snapshot = SystemSnapshot(timestamp=time.time())
        completed = [
            RequestMetrics(seq_id=0, arrival_time=1.0, first_token_time=1.1,
                          finish_time=1.5, num_prompt_tokens=10, num_output_tokens=5),
        ]
        exporter.update(snapshot, new_completed=completed)


# =============================================================================
# Part 3: Monitor Integration
# =============================================================================

class TestPart3:
    """Tests for Monitor (combined collector + exporter)."""

    def test_monitor_init(self):
        from vkv.engine.monitor import Monitor
        monitor = Monitor(enable_prometheus=False)
        assert monitor.collector is not None

    def test_monitor_request_lifecycle(self):
        from vkv.engine.monitor import Monitor
        monitor = Monitor(enable_prometheus=False)

        monitor.on_request_arrival(seq_id=0, num_prompt_tokens=10)
        monitor.on_first_token(seq_id=0)
        monitor.on_request_finish(seq_id=0, num_output_tokens=5)

        m = monitor.collector.get_request_metrics(0)
        assert m.ttft > 0
        assert m.num_output_tokens == 5

    def test_monitor_on_step(self):
        from vkv.engine.monitor import Monitor
        from vkv.engine.block_manager import BlockManager
        from vkv.engine.scheduler import Scheduler, SchedulerConfig

        mgr = BlockManager(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=100, num_cpu_blocks=50), device="cpu")
        sched = Scheduler(mgr, SchedulerConfig())
        monitor = Monitor(enable_prometheus=False)

        snapshot = monitor.on_step(mgr, sched)
        assert snapshot.total_blocks == 100
        assert snapshot.free_blocks == 100

    def test_monitor_with_engine(self):
        """Integration test: Monitor inside LLMEngine loop."""
        from vkv.engine.monitor import Monitor
        from vkv.engine.block_manager import BlockManager
        from vkv.engine.scheduler import Scheduler, SchedulerConfig
        from vkv.engine.sequence import Sequence

        mgr = BlockManager(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=100, num_cpu_blocks=50), device="cpu")
        sched = Scheduler(mgr, SchedulerConfig())
        monitor = Monitor(enable_prometheus=False)

        seq = Sequence(list(range(10)), mgr, SamplingParams(max_tokens=3))
        sched.add(seq)
        monitor.on_request_arrival(seq.seq_id, seq.num_prompt_tokens)

        sched.schedule()  # prefill
        monitor.on_first_token(seq.seq_id)

        snapshot = monitor.on_step(mgr, sched)
        assert snapshot.num_running == 1
        assert snapshot.used_blocks > 0
