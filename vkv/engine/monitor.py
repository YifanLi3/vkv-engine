"""
Phase 5: Monitoring & Analysis

Collects real-time metrics from BlockManager, Scheduler, and LLMEngine.
Exposes them via Prometheus for Grafana dashboards.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server, REGISTRY, CollectorRegistry


# =============================================================================
# Part 1: MetricsCollector
# =============================================================================

@dataclass
class RequestMetrics:
    """Timing metrics for a single request. Provided for you."""
    seq_id: int
    arrival_time: float = 0.0
    first_token_time: float = 0.0
    finish_time: float = 0.0
    num_prompt_tokens: int = 0
    num_output_tokens: int = 0

    @property
    def ttft(self) -> float:
        """Time To First Token (seconds)."""
        if self.first_token_time > 0 and self.arrival_time > 0:
            return self.first_token_time - self.arrival_time
        return 0.0

    @property
    def tpot(self) -> float:
        """Average Time Per Output Token (seconds)."""
        if self.num_output_tokens > 0 and self.finish_time > self.first_token_time:
            return (self.finish_time - self.first_token_time) / self.num_output_tokens
        return 0.0

    @property
    def total_time(self) -> float:
        """Total request latency (seconds)."""
        if self.finish_time > 0 and self.arrival_time > 0:
            return self.finish_time - self.arrival_time
        return 0.0


@dataclass
class SystemSnapshot:
    """Point-in-time snapshot of system state. Provided for you."""
    timestamp: float = 0.0
    total_blocks: int = 0
    used_blocks: int = 0
    free_blocks: int = 0
    utilization: float = 0.0
    num_waiting: int = 0
    num_running: int = 0
    num_swapped: int = 0
    num_preemptions: int = 0
    num_completed: int = 0


class MetricsCollector:
    """
    Collects metrics from BlockManager and Scheduler.

    Usage:
        >>> collector = MetricsCollector()
        >>> collector.on_request_arrival(seq_id=0, num_prompt_tokens=50)
        >>> collector.on_first_token(seq_id=0)
        >>> collector.on_request_finish(seq_id=0, num_output_tokens=30)
        >>> snapshot = collector.collect_snapshot(block_manager, scheduler)
    """

    def __init__(self):
        """
        Initialize the metrics collector.

        1. self._request_metrics: Dict[int, RequestMetrics] = {}
        2. self._completed_metrics: List[RequestMetrics] = []
        3. self._num_preemptions: int = 0
        4. self._num_completed: int = 0
        """
        self._request_metrics: Dict[int, RequestMetrics] = {}
        self._completed_metrics: List[RequestMetrics] = []
        self._num_preemptions: int = 0
        self._num_completed: int = 0

    def on_request_arrival(self, seq_id: int, num_prompt_tokens: int) -> None:
        """
        Record when a new request arrives.

        Create a RequestMetrics with arrival_time = time.time()
        """
        self._request_metrics[seq_id] = RequestMetrics(
            seq_id=seq_id,
            arrival_time=time.time(),
            num_prompt_tokens=num_prompt_tokens,
        )

    def on_first_token(self, seq_id: int) -> None:
        """
        Record when the first token is generated (after prefill).

        Set first_token_time = time.time()
        """
        metrics = self._request_metrics[seq_id]
        metrics.first_token_time = time.time()

    def on_request_finish(self, seq_id: int, num_output_tokens: int) -> None:
        """
        Record when a request finishes.

        1. Set finish_time = time.time() and num_output_tokens
        2. Move from _request_metrics to _completed_metrics
        3. Increment _num_completed
        """
        metrics = self._request_metrics[seq_id]
        metrics.finish_time = time.time()
        metrics.num_output_tokens = num_output_tokens

        self._completed_metrics.append(metrics)
        del self._request_metrics[seq_id]
        self._num_completed += 1

    def on_preemption(self) -> None:
        """Record a preemption event.

        """
        self._num_preemptions += 1


    def collect_snapshot(self, block_manager, scheduler) -> SystemSnapshot:
        """
        Collect a point-in-time snapshot of the system.

        Args:
            block_manager: BlockManager instance
            scheduler: Scheduler instance

        Returns:
            SystemSnapshot with current stats

        Read stats from block_manager.stats and scheduler queue lengths.
        """
        stats = block_manager.stats

        return SystemSnapshot(
            timestamp=time.time(),
            total_blocks=stats.total_blocks,
            used_blocks=stats.used_blocks,
            free_blocks=stats.free_blocks,
            utilization=stats.utilization,
            num_waiting=len(scheduler.waiting),
            num_running=len(scheduler.running),
            num_swapped=len(scheduler.swapped),
            num_preemptions=self._num_preemptions,
            num_completed=self._num_completed,
        )

    def get_request_metrics(self, seq_id: int) -> Optional[RequestMetrics]:
        """Get metrics for a specific request (active or completed)."""
        if seq_id in self._request_metrics:
            return self._request_metrics[seq_id]
        for m in self._completed_metrics:
            if m.seq_id == seq_id:
                return m
        return None

    def get_summary(self) -> dict:
        """
        Get summary statistics across all completed requests.

        Returns:
            Dict with avg_ttft, avg_tpot, p99_ttft, throughput, etc.

        """
        if not self._completed_metrics:
            return {
                "num_completed": 0,
                "avg_ttft": 0.0,
                "avg_tpot": 0.0,
                "p99_ttft": 0.0,
                "throughput_tokens_per_sec": 0.0,
                "total_output_tokens": 0,
            }

        ttfts = [m.ttft for m in self._completed_metrics]
        tpots = [m.tpot for m in self._completed_metrics if m.tpot > 0]
        total_tokens = sum(m.num_output_tokens for m in self._completed_metrics)
        total_time = sum(m.total_time for m in self._completed_metrics)

        sorted_ttfts = sorted(ttfts)
        p99_idx = min(int(math.ceil(len(sorted_ttfts) * 0.99)), len(sorted_ttfts)) - 1
        p99_ttft = sorted_ttfts[p99_idx]

        return {
            "num_completed": self._num_completed,
            "avg_ttft": sum(ttfts) / len(ttfts),
            "avg_tpot": sum(tpots) / len(tpots) if tpots else 0.0,
            "p99_ttft": p99_ttft,
            "throughput_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0.0,
            "total_output_tokens": total_tokens,
        }


# =============================================================================
# Part 2: Prometheus Exporter
# =============================================================================

class PrometheusExporter:
    """
    Exposes metrics via Prometheus HTTP endpoint.

    Usage:
        >>> exporter = PrometheusExporter(port=9090)
        >>> exporter.update(snapshot, completed_metrics)
    """

    def __init__(self, port: int = 9090):
        """
        Define Prometheus metrics and start HTTP server.

        1. Define Gauge/Counter/Histogram metrics
        2. Start HTTP server (optional in tests, required in production)
        """
        self.port = port

        # Use a separate registry to avoid duplicate registration in tests
        self._registry = CollectorRegistry()

        # Block metrics
        self.block_utilization = Gauge('vkv_block_utilization', 'Block pool utilization ratio', registry=self._registry)
        self.free_blocks = Gauge('vkv_free_blocks', 'Number of free GPU blocks', registry=self._registry)
        self.used_blocks = Gauge('vkv_used_blocks', 'Number of used GPU blocks', registry=self._registry)

        # Queue metrics
        self.num_waiting = Gauge('vkv_num_waiting', 'Sequences in waiting queue', registry=self._registry)
        self.num_running = Gauge('vkv_num_running', 'Sequences in running queue', registry=self._registry)
        self.num_swapped = Gauge('vkv_num_swapped', 'Sequences in swapped queue', registry=self._registry)

        # Request metrics
        self.ttft_histogram = Histogram(
            'vkv_ttft_seconds', 'Time to First Token',
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            registry=self._registry,
        )
        self.tpot_histogram = Histogram(
            'vkv_tpot_seconds', 'Time per Output Token',
            buckets=[0.005, 0.01, 0.02, 0.05, 0.1],
            registry=self._registry,
        )

        # Counters
        self.requests_completed = Counter('vkv_requests_completed_total', 'Total completed requests', registry=self._registry)
        self.preemptions_total = Counter('vkv_preemptions_total', 'Total preemptions', registry=self._registry)
        self.tokens_generated = Counter('vkv_tokens_generated_total', 'Total tokens generated', registry=self._registry)

    def update(self, snapshot: SystemSnapshot, new_completed: List[RequestMetrics] = None) -> None:
        """
        Update all Prometheus metrics from a snapshot.

        Args:
            snapshot: Current system state
            new_completed: Newly completed request metrics (for histograms)

        1. Set gauge values from snapshot
        2. Observe TTFT/TPOT histograms for new completed requests
        3. Increment counters
        """
        self.block_utilization.set(snapshot.utilization)
        self.free_blocks.set(snapshot.free_blocks)
        self.used_blocks.set(snapshot.used_blocks)
        self.num_waiting.set(snapshot.num_waiting)
        self.num_running.set(snapshot.num_running)
        self.num_swapped.set(snapshot.num_swapped)
        self.preemptions_total._value.set(snapshot.num_preemptions)

        if new_completed:
            for m in new_completed:
                if m.ttft > 0:
                    self.ttft_histogram.observe(m.ttft)
                if m.tpot > 0:
                    self.tpot_histogram.observe(m.tpot)

                self.requests_completed.inc()
                self.tokens_generated.inc(m.num_output_tokens)

    def start_server(self):
        """Start the Prometheus HTTP server. Call once at startup."""
        start_http_server(self.port)


# =============================================================================
# Part 3: Integration helpers
# =============================================================================

class Monitor:
    """
    Combines MetricsCollector + PrometheusExporter.
    Convenience class for LLMEngine integration.

    Usage in LLMEngine.step():
        self.monitor.on_step(block_manager, scheduler, finished_seqs)
    """

    def __init__(self, port: int = 9090, enable_prometheus: bool = False):
        """
        1. Create MetricsCollector
        2. Create PrometheusExporter (if enabled)
        3. Optionally start HTTP server
        """
        self.

    def on_request_arrival(self, seq_id: int, num_prompt_tokens: int) -> None:
        """Delegate to collector.

        TODO: Implement this.
        """
        raise NotImplementedError("TODO")

    def on_first_token(self, seq_id: int) -> None:
        """Delegate to collector.

        TODO: Implement this.
        """
        raise NotImplementedError("TODO")

    def on_request_finish(self, seq_id: int, num_output_tokens: int) -> None:
        """Delegate to collector.

        TODO: Implement this.
        """
        raise NotImplementedError("TODO")

    def on_preemption(self) -> None:
        """Delegate to collector.

        TODO: Implement this.
        """
        raise NotImplementedError("TODO")

    def on_step(self, block_manager, scheduler, finished_seqs=None) -> SystemSnapshot:
        """
        Called once per LLMEngine.step().
        Collect snapshot, update Prometheus, return snapshot.

        TODO: Implement this.
        1. collector.collect_snapshot(block_manager, scheduler)
        2. If prometheus enabled, exporter.update(snapshot)
        3. Return snapshot
        """
        raise NotImplementedError("TODO: Implement Monitor.on_step")

    def get_summary(self) -> dict:
        """Get summary stats. Delegate to collector."""
        return self.collector.get_summary()
