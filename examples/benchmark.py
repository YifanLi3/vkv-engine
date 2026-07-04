"""
Phase 6, Part 5: Benchmark — HF default vs vkv-engine

Compares:
  - HuggingFace default generate() with DynamicCache
  - vkv-engine RealModelRunner with PagedCache + BlockManager

Metrics:
  - Throughput (tokens/s)
  - Peak GPU memory (GB)
  - Max concurrent requests before OOM
"""

import logging
import time
import warnings
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

from vkv.config import ModelConfig, CacheConfig
from vkv.engine.block_manager import BlockManager
from vkv.engine.real_model_runner import RealModelRunner
from vkv.sampling_params import SamplingParams

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda"
MAX_NEW_TOKENS = 100
NUM_RUNS = 3  # Average over multiple runs


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def reset_gpu_stats():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def get_peak_memory_gb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def measure_throughput(fn, num_runs: int = NUM_RUNS) -> tuple[float, float]:
    """
    Run fn() num_runs times and return (avg_tokens_per_sec, peak_memory_gb).
    fn() should return the number of generated tokens.
    """
    total_tokens = 0
    total_time = 0.0
    reset_gpu_stats()

    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        n_tokens = fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_tokens += n_tokens
        total_time += elapsed

    peak_mem = get_peak_memory_gb()
    throughput = total_tokens / total_time
    return throughput, peak_mem


# ─────────────────────────────────────────────────────────
# Task 5.1: Single-request throughput comparison
# ─────────────────────────────────────────────────────────

def benchmark_hf_default(model, tokenizer, prompt: str) -> tuple[float, float]:
    """Benchmark HuggingFace default generate() with DynamicCache."""

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    def run():
        # Return number of generated tokens
        output = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS)
        return output.shape[1] - input_ids.shape[1]

    return measure_throughput(run)


def benchmark_vkv_engine(runner, prompt: str) -> tuple[float, float]:
    """Benchmark vkv-engine generate() with PagedCache."""

    def run():
        output = runner.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
        return len(runner.tokenizer.encode(output))

    return measure_throughput(run)


# ─────────────────────────────────────────────────────────
# Task 5.2: Concurrent requests — throughput at each concurrency level
# ─────────────────────────────────────────────────────────

BATCH_MAX_TOKENS = 50


def benchmark_concurrent_hf(model, tokenizer, prompt: str, batch_size: int) -> tuple[float, float]:
    """Return (tokens/s, peak_mem_GB) when serving `batch_size` requests in parallel via HF batching."""
    prompts = [prompt] * batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
    reset_gpu_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=BATCH_MAX_TOKENS)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    n_generated = (output.shape[1] - inputs["input_ids"].shape[1]) * batch_size
    return n_generated / elapsed, get_peak_memory_gb()


def benchmark_concurrent_vkv(engine, tokenizer, prompt: str, batch_size: int) -> tuple[float, float]:
    """Return (tokens/s, peak_mem_GB) when serving `batch_size` requests in parallel via vkv-engine."""
    sp = SamplingParams(max_tokens=BATCH_MAX_TOKENS)
    prompts = [tokenizer.encode(prompt)] * batch_size
    reset_gpu_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = engine.generate(prompts=prompts, sampling_params=sp)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    n_generated = sum(len(o.output_token_ids) for o in outputs)
    return n_generated / elapsed, get_peak_memory_gb()


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    prompt = "What is artificial intelligence and how does it work?"

    print("=" * 60)
    print("Loading model and tokenizer...")
    print("=" * 60)

    # Load shared model for HF baseline
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    # Setup vkv-engine runner
    model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
    cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=500, num_cpu_blocks=50)
    block_manager = BlockManager(model_cfg, cache_cfg, device=DEVICE)
    runner = RealModelRunner(MODEL_NAME, block_manager, device=DEVICE)

    print("\n" + "=" * 60)
    print("Task 5.1: Single-request Throughput")
    print("=" * 60)

    hf_throughput, hf_peak_mem = benchmark_hf_default(hf_model, tokenizer, prompt)
    print(f"HF default:   {hf_throughput:6.1f} tokens/s  |  peak mem: {hf_peak_mem:.2f} GB")

    vkv_throughput, vkv_peak_mem = benchmark_vkv_engine(runner, prompt)
    print(f"vkv-engine:   {vkv_throughput:6.1f} tokens/s  |  peak mem: {vkv_peak_mem:.2f} GB")

    speedup = vkv_throughput / hf_throughput
    print(f"\nSpeedup: {speedup:.2f}x")

    print("\n" + "=" * 60)
    print("Task 5.2: Concurrent Throughput (tokens/s at each batch size)")
    print("=" * 60)

    from vkv.engine.real_llm_engine import RealLLMEngine
    from vkv.engine.scheduler import SchedulerConfig

    scheduler_cfg = SchedulerConfig(max_num_seqs=32)
    engine = RealLLMEngine(MODEL_NAME, model_cfg, cache_cfg, scheduler_cfg, device=DEVICE)

    print(f"{'Batch':>6} | {'HF tok/s':>10} {'HF mem':>8} | {'vkv tok/s':>10} {'vkv mem':>8} | speedup")
    print("-" * 68)
    for batch in [1, 2, 4, 8, 16]:
        try:
            hf_tps, hf_mem = benchmark_concurrent_hf(hf_model, tokenizer, prompt, batch)
        except Exception as e:
            print(f"{batch:>6} | HF OOM: {e}")
            hf_tps = hf_mem = 0
        try:
            vkv_tps, vkv_mem = benchmark_concurrent_vkv(engine, tokenizer, prompt, batch)
        except Exception as e:
            print(f"{batch:>6} | vkv OOM: {e}")
            vkv_tps = vkv_mem = 0
        speedup = vkv_tps / hf_tps if hf_tps > 0 else 0
        print(f"{batch:>6} | {hf_tps:>10.1f} {hf_mem:>7.2f}G | {vkv_tps:>10.1f} {vkv_mem:>7.2f}G | {speedup:>5.2f}x")


if __name__ == "__main__":
    main()
