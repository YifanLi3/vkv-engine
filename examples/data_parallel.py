"""
Phase 6 Extension: Data Parallelism across multiple GPUs

Each GPU runs an independent RealLLMEngine instance.
Requests are distributed round-robin across GPUs.

Requires >= 2 GPUs. Run with: uv run python examples/data_parallel.py
"""

import os
import torch
import torch.multiprocessing as mp
from typing import List

from vkv.config import ModelConfig, CacheConfig
from vkv.engine.scheduler import SchedulerConfig
from vkv.sampling_params import SamplingParams

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 30


def worker(gpu_id: int, prompts_chunk: List[str], result_queue: mp.Queue):
    """
    Runs on one GPU. Loads its own model and processes a chunk of prompts.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch

    # Delayed import so CUDA_VISIBLE_DEVICES takes effect
    from multi_inference import RealLLMEngine
    from transformers import AutoTokenizer

    print(f"[GPU {gpu_id}] loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
    cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=300, num_cpu_blocks=20)
    scheduler_cfg = SchedulerConfig(max_num_seqs=8)

    engine = RealLLMEngine(
        MODEL_NAME, model_cfg, cache_cfg, scheduler_cfg, device="cuda",
    )
    sp = SamplingParams(max_tokens=MAX_NEW_TOKENS)

    print(f"[GPU {gpu_id}] processing {len(prompts_chunk)} prompts...")
    outputs = engine.generate(
        prompts=[tokenizer.encode(p) for p in prompts_chunk],
        sampling_params=sp,
    )

    results = [(gpu_id, prompt, tokenizer.decode(out.output_token_ids, skip_special_tokens=True))
               for prompt, out in zip(prompts_chunk, outputs)]
    result_queue.put(results)


def main():
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    if num_gpus < 2:
        print("Warning: only 1 GPU visible. DP demo needs >= 2 GPUs.")
        print("Running on a single GPU for demonstration...")
        num_gpus = 1

    all_prompts = [
        "What is AI?",
        "Explain neural networks.",
        "What is KV cache?",
        "Describe transformer models.",
        "How does attention work?",
        "What is quantization?",
        "Explain paged attention.",
        "What is inference optimization?",
    ]

    chunks = [all_prompts[i::num_gpus] for i in range(num_gpus)]

    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker, args=(gpu_id, chunks[gpu_id], result_queue))
        p.start()
        processes.append(p)

    all_results = []
    for _ in range(num_gpus):
        all_results.extend(result_queue.get())

    for p in processes:
        p.join()

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for gpu_id, prompt, output in sorted(all_results):
        print(f"[GPU {gpu_id}] {prompt}")
        print(f"           → {output}\n")


if __name__ == "__main__":
    main()
