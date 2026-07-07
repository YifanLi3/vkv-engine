# Phase 5: Monitoring & Analysis Dashboard

> **Est. time**: 6–8 hours | **Difficulty**: ★★☆☆☆
> **Prerequisites**: Phases 1–2 done
> **Requires GPU**: No

---

## Table of Contents

- [Part 0: Background — Why monitoring matters](#part-0-background)
- [Part 1: MetricsCollector — Metric collection (core)](#part-1-collector)
- [Part 2: Prometheus Exporter (core)](#part-2-prometheus)
- [Part 3: Integrate with LLMEngine (core)](#part-3-integration)
- [Part 4: Dashboard configuration (advanced)](#part-4-dashboard)

---

<a id="part-0-background"></a>
## Part 0: Background

A production system needs to answer questions like:
- How much GPU memory is used? How many more requests can it serve?
- What are the TTFT / TPOT of each request? Does p99 latency meet SLO?
- What is the prefix-cache hit rate? Is it worth enabling?
- Is fragmentation high? Do we need to retune `block_size`?

Answering these requires **real-time metrics** — you can't guess.

### Prometheus + Grafana architecture

```
vkv-engine                    Prometheus              Grafana
┌────────────────────┐        ┌──────────┐           ┌──────────┐
│ MetricsCollector   │─export→│ scrape   │──query──→│ visualize │
│ (port 9090)        │        │ + TS DB  │           │ dashboard │
└────────────────────┘        └──────────┘           └──────────┘
```

---

<a id="part-1-collector"></a>
## Part 1: MetricsCollector — Metric collection [core]

> **File**: `vkv/engine/monitor.py`
> **Tests**: `uv run pytest tests/test_phase5.py -k "part1" -v`

Collect statistics from `BlockManager` and `Scheduler`.

### Task 1.1: Implement `MetricsCollector.__init__()`
### Task 1.2: Implement `MetricsCollector.collect()`

Gather current state from each component:
- BlockManager: `used_blocks`, `free_blocks`, `utilization`
- Scheduler: lengths of waiting / running / swapped queues
- Per-request: TTFT, TPOT

### Task 1.3: Implement `MetricsCollector.record_request_metrics()`

Record per-request latency metrics.

---

<a id="part-2-prometheus"></a>
## Part 2: Prometheus Exporter [core]

> **File**: `vkv/engine/monitor.py`
> **Tests**: `uv run pytest tests/test_phase5.py -k "part2" -v`

### Task 2.1: Define Prometheus metrics
### Task 2.2: Implement `update_prometheus_metrics()`

---

<a id="part-3-integration"></a>
## Part 3: Integrate with LLMEngine [core]

> **File**: `vkv/engine/monitor.py` + `vkv/engine/llm_engine.py`
> **Tests**: `uv run pytest tests/test_phase5.py -k "part3" -v`

Collect metrics at every `LLMEngine.step()`.

---

<a id="part-4-dashboard"></a>
## Part 4: Dashboard configuration [advanced]

Generate a Grafana dashboard JSON.

---

## Running the tests

```bash
uv run pytest tests/test_phase5.py -v
uv run pytest tests/test_phase5.py -k "part1" -v
```
