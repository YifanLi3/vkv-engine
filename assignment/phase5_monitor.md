# Phase 5: Monitoring & Analysis Dashboard

> **预计用时**: 6-8 小时 | **难度**: ★★☆☆☆  
> **前置知识**: Phase 1-2 完成  
> **需要 GPU**: 否

---

## 目录

- [Part 0: Background — 为什么需要监控](#part-0-background)
- [Part 1: MetricsCollector — 指标采集 (核心)](#part-1-collector)
- [Part 2: Prometheus Exporter (核心)](#part-2-prometheus)
- [Part 3: 集成到 LLMEngine (核心)](#part-3-integration)
- [Part 4: Dashboard 配置 (进阶)](#part-4-dashboard)

---

<a id="part-0-background"></a>
## Part 0: Background

生产系统需要回答这些问题：
- GPU 显存用了多少？还能服务几个请求？
- 请求的 TTFT/TPOT 是多少？p99 延迟是否达标？
- Prefix cache 命中率如何？值不值得开？
- 碎片率高不高？需不需要调 block_size？

这些问题需要**实时指标**来回答，不能靠猜。

### Prometheus + Grafana 架构

```
vkv-engine                    Prometheus              Grafana
┌──────────────┐             ┌──────────┐           ┌──────────┐
│ MetricsCollector │──export──→│ 拉取指标  │──query──→│ 可视化    │
│ (port 9090)  │             │ 存储时序  │           │ Dashboard │
└──────────────┘             └──────────┘           └──────────┘
```

---

<a id="part-1-collector"></a>
## Part 1: MetricsCollector — 指标采集 [核心]

> **文件**: `vkv/engine/monitor.py`  
> **测试**: `uv run pytest tests/test_phase5.py -k "part1" -v`

从 BlockManager 和 Scheduler 收集统计数据。

### Task 1.1: 实现 `MetricsCollector.__init__()`
### Task 1.2: 实现 `MetricsCollector.collect()`

从各组件收集当前状态：
- BlockManager: used_blocks, free_blocks, utilization
- Scheduler: waiting/running/swapped 队列长度
- 请求级: TTFT, TPOT

### Task 1.3: 实现 `MetricsCollector.record_request_metrics()`

记录每个请求的延迟指标。

---

<a id="part-2-prometheus"></a>
## Part 2: Prometheus Exporter [核心]

> **文件**: `vkv/engine/monitor.py`  
> **测试**: `uv run pytest tests/test_phase5.py -k "part2" -v`

### Task 2.1: 定义 Prometheus metrics
### Task 2.2: 实现 `update_prometheus_metrics()`

---

<a id="part-3-integration"></a>
## Part 3: 集成到 LLMEngine [核心]

> **文件**: `vkv/engine/monitor.py` + `vkv/engine/llm_engine.py`  
> **测试**: `uv run pytest tests/test_phase5.py -k "part3" -v`

在 LLMEngine.step() 里每步收集指标。

---

<a id="part-4-dashboard"></a>
## Part 4: Dashboard 配置 [进阶]

生成 Grafana dashboard JSON。

---

## 运行测试

```bash
uv run pytest tests/test_phase5.py -v
uv run pytest tests/test_phase5.py -k "part1" -v
```
