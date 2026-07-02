"""
Phase 6, Part 4, Task 4.1: Single-request inference

End-to-end inference using RealModelRunner.generate() directly.
"""

import torch
from vkv.config import ModelConfig, CacheConfig
from vkv.engine.block_manager import BlockManager
from vkv.engine.real_model_runner import RealModelRunner

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda"

def main():
    # Task 4.1.1: Initialize ModelConfig and CacheConfig
    # TinyLlama: num_layers=22, num_kv_heads=4, head_dim=64
    model_cfg = ModelConfig(
        # TODO: fill in TinyLlama parameters
    )
    cache_cfg = CacheConfig(
        # TODO: choose appropriate block_size and num_gpu_blocks
    )

    # Task 4.1.2: Create BlockManager and RealModelRunner
    block_manager = BlockManager(
        # TODO
    )
    runner = RealModelRunner(
        # TODO
    )

    # Task 4.1.3: Call generate() to produce text
    prompts = [
        "What is artificial intelligence?",
        "Explain KV cache in one sentence.",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        output = runner.generate(
            prompt=prompt,
            max_new_tokens=50,
            temperature=1.0,
        )
        print(f"Output: {output}")


if __name__ == "__main__":
    main()
