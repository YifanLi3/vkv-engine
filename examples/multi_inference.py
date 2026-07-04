"""
Phase 6, Part 4, Task 4.2: Multi-request concurrent inference demo.

Uses RealLLMEngine (defined in vkv.engine.real_llm_engine) to run
multiple prompts concurrently on a single GPU.
"""

from vkv.config import ModelConfig, CacheConfig
from vkv.engine.real_llm_engine import RealLLMEngine
from vkv.engine.scheduler import SchedulerConfig
from vkv.sampling_params import SamplingParams

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda"


def main():
    model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
    cache_cfg = CacheConfig(num_gpu_blocks=500, num_cpu_blocks=50)
    scheduler_cfg = SchedulerConfig()

    engine = RealLLMEngine(
        model_name=MODEL_NAME,
        model_config=model_cfg,
        cache_config=cache_cfg,
        scheduler_config=scheduler_cfg,
        device=DEVICE,
    )
    tokenizer = engine.model_runner.tokenizer

    prompts = [
        "What is AI?",
        "Explain neural networks.",
        "What is KV cache?",
    ]

    sp = SamplingParams(max_tokens=30)
    outputs = engine.generate(
        prompts=[tokenizer.encode(p) for p in prompts],
        sampling_params=sp,
    )

    for prompt, out in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Output: {tokenizer.decode(out.output_token_ids, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
