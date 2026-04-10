# vkv-engine

An industrial-grade KV Cache management engine for LLM inference, inspired by [vLLM](https://github.com/vllm-project/vllm) PagedAttention and [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm).

## What is this?

A hands-on, assignment-style project that teaches you how LLM serving systems manage GPU memory for KV Cache — the core bottleneck in production LLM inference.

All class names are aligned with **nano-vLLM / vLLM** (`Block`, `BlockManager`, `Sequence`, `Scheduler`, ...) so you can seamlessly transition to reading production codebases.

## Project Focus

```
┌─────────────────────┐
│  LLMEngine / API    │
│  Tokenizer          │
└────────┬────────────┘
┌────────▼────────────┐
│  Scheduler          │  ← Phase 2
└────────┬────────────┘
┌────────▼────────────┐
│  ModelRunner        │
└────────┬────────────┘
┌════════▼════════════┐
║  BlockManager       ║
║  BlockAllocator     ║  ← ★ Core focus ★
║  Sequence           ║
║  Swapper / Evictor  ║
║  Quantizer          ║
║  Prefix Cache       ║
╚═════════════════════╝
```

## Phased Implementation

| Phase | Topic | Duration |
|-------|-------|----------|
| **Phase 1** | Block-based Memory Management (BlockManager, Sequence, Swap, Eviction) | 2-3 weeks |
| **Phase 2** | Continuous Batching Scheduler | 2-3 weeks |
| **Phase 3** | KV Cache Quantization (INT8/INT4) | 2 weeks |
| **Phase 4** | Prefix Caching (Radix Tree + COW) | 2 weeks |
| **Phase 5** | Monitoring Dashboard (Prometheus + Grafana) | 1-2 weeks |

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
git clone https://github.com/<YOUR_USERNAME>/vkv-engine.git
cd vkv-engine
uv sync
```

### How to Work on It

1. Read the assignment doc: `assignment/phase1.md`
2. Implement the TODOs in `vkv/engine/*.py` (look for `raise NotImplementedError`)
3. Run tests to verify:

```bash
# Run all Phase 1 tests
uv run pytest tests/test_phase1.py -v

# Run a specific part
uv run pytest tests/test_phase1.py -k "part1" -v
uv run pytest tests/test_phase1.py -k "part2" -v
# ...
```

## Naming Alignment

| vkv-engine | nano-vLLM | vLLM |
|------------|-----------|------|
| `Block` | `Block` | `PhysicalTokenBlock` |
| `BlockAllocator` | (inline) | `BlockAllocator` |
| `BlockManager` | `BlockManager` | `BlockSpaceManager` |
| `Sequence` | `Sequence` | `Sequence` |
| `SequenceStatus` | `SequenceStatus` | `SequenceStatus` |
| `Scheduler` | `Scheduler` | `Scheduler` |
| `SamplingParams` | `SamplingParams` | `SamplingParams` |

## References

- [vLLM: PagedAttention](https://arxiv.org/abs/2309.06180)
- [nano-vLLM (~1200 lines)](https://github.com/GeeeekExplorer/nano-vllm)
- [Orca: Continuous Batching](https://www.usenix.org/conference/osdi22/presentation/yu)
- [SGLang: RadixAttention](https://arxiv.org/abs/2312.07104)
