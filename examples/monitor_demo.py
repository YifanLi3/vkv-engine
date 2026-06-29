"""
Demo: Run vkv-engine with Prometheus monitoring.

Usage:
    1. Start Prometheus + Grafana:
       cd monitoring && docker compose up -d

    2. Run this script:
       uv run python examples/monitor_demo.py

    3. Open Grafana: http://localhost:3000 (admin/admin)
       Add data source → Prometheus → URL: http://prometheus:9091
       Create dashboard → Add panel → Query: vkv_block_utilization
"""

import time

from vkv.config import TINY_MODEL, CacheConfig
from vkv.sampling_params import SamplingParams
from vkv.engine.block_manager import BlockManager
from vkv.engine.scheduler import Scheduler, SchedulerConfig
from vkv.engine.sequence import Sequence
from vkv.engine.monitor import Monitor

# vkv-engine exposes metrics on port 9090
monitor = Monitor(port=9090, enable_prometheus=True)

# Setup
block_manager = BlockManager(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=50, num_cpu_blocks=20), device="cpu")
scheduler = Scheduler(block_manager, SchedulerConfig(max_num_seqs=8))

print("vkv-engine metrics server running on http://localhost:9090")
print("Open Grafana at http://localhost:3000")
print("Simulating requests...\n")

seq_counter = 0

while True:
    # Add a new request every 2 seconds
    token_ids = list(range(10 + seq_counter * 3))
    seq = Sequence(token_ids, block_manager, SamplingParams(max_tokens=10))
    scheduler.add(seq)
    monitor.on_request_arrival(seq.seq_id, seq.num_prompt_tokens)
    seq_counter += 1

    # Run a few steps
    for _ in range(5):
        output = scheduler.schedule()

        if output.is_prefill:
            for s in output.scheduled_seqs:
                monitor.on_first_token(s.seq_id)
        elif output.scheduled_seqs:
            # Only postprocess seqs that have blocks allocated
            decode_seqs = [s for s in output.scheduled_seqs if s.block_table]
            if decode_seqs:
                token_ids_gen = [42] * len(decode_seqs)
                finished = scheduler.postprocess(decode_seqs, token_ids_gen)
                for s in finished:
                    monitor.on_request_finish(s.seq_id, s.num_tokens - s.num_prompt_tokens)

        snapshot = monitor.on_step(block_manager, scheduler)
        print(f"  blocks: {snapshot.used_blocks}/{snapshot.total_blocks} "
              f"waiting: {snapshot.num_waiting} running: {snapshot.num_running}")

        time.sleep(0.5)

    summary = monitor.get_summary()
    print(f"\n  Summary: completed={summary['num_completed']} "
          f"avg_ttft={summary['avg_ttft']:.4f}s "
          f"avg_tpot={summary['avg_tpot']:.4f}s\n")

    time.sleep(1)
