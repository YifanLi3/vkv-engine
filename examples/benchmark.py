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

import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        # TODO: call model.generate() with default settings (no past_key_values override)
        # Return number of generated tokens
        output = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS)
        return output.shape[1] - input_ids.shape[1]

    return measure_throughput(run)


def benchmark_vkv_engine(runner, prompt: str) -> tuple[float, float]:
    """Benchmark vkv-engine generate() with PagedCache."""

    def run():
        # TODO: call runner.generate() and return number of generated tokens
        output = runner.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
        return len(runner.tokenizer.encode(output))

    return measure_throughput(run)


# ─────────────────────────────────────────────────────────
# Task 5.2: Concurrent requests — max before OOM
# ─────────────────────────────────────────────────────────

def benchmark_max_concurrent_hf(model, tokenizer, prompt: str) -> int:
    """
    Find max concurrent requests HF can handle before OOM.

    Strategy: batch multiple prompts together (batch_size > 1).
    Increase batch_size until OOM.
    """
    # TODO: implement batch inference with increasing batch size
    # Hint: tokenizer(prompts, return_tensors="pt", padding=True) for batching
    max_batch = 0
    for batch_size in range(1, 20):
        try:
            prompts = [prompt] * batch_size
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
            model.generate(**inputs, max_new_tokens=50)
            max_batch = batch_size
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            break
    return max_batch


def benchmark_max_concurrent_vkv(model_cfg, cache_cfg, tokenizer, prompt: str) -> int:
    """
    Find max concurrent requests vkv-engine can handle before OOM.

    Strategy: submit increasing number of requests to RealLLMEngine.
    """
    # TODO: use RealLLMEngine from multi_inference.py
    # Increase number of prompts until OOM or block exhaustion
    from examples.multi_inference import RealLLMEngine
    from vkv.engine.scheduler import SchedulerConfig

    max_concurrent = 0
    for n in range(1, 20):
        try:
            scheduler_cfg = SchedulerConfig(max_num_seqs=n)
            engine = RealLLMEngine(MODEL_NAME, model_cfg, cache_cfg, scheduler_cfg, device=DEVICE)
            sp = SamplingParams(max_tokens=50)
            engine.generate(
                prompts=[tokenizer.encode(prompt)] * n,
                sampling_params=sp,
            )
            max_concurrent = n
            del engine
            torch.cuda.empty_cache()
        except (torch.cuda.OutOfMemoryError, Exception):
            break
    return max_concurrent


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
    print("Task 5.2: Max Concurrent Requests Before OOM")
    print("=" * 60)

    max_hf = benchmark_max_concurrent_hf(hf_model, tokenizer, prompt)
    print(f"HF default:   max {max_hf} concurrent requests")

    max_vkv = benchmark_max_concurrent_vkv(model_cfg, cache_cfg, tokenizer, prompt)
    print(f"vkv-engine:   max {max_vkv} concurrent requests")


if __name__ == "__main__":
    main()
