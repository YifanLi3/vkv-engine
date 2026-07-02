"""
Phase 6, Part 4, Task 4.1: 单条推理

直接使用 RealModelRunner.generate() 跑端到端推理。
"""

import torch
from vkv.config import ModelConfig, CacheConfig
from vkv.engine.block_manager import BlockManager
from vkv.engine.real_model_runner import RealModelRunner

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda"

def main():
    # Task 4.1.1: 初始化 ModelConfig 和 CacheConfig
    # TinyLlama: num_layers=22, num_kv_heads=4, head_dim=64
    model_cfg = ModelConfig(
        # TODO: 填入 TinyLlama 的参数
    )
    cache_cfg = CacheConfig(
        # TODO: 选择合适的 block_size 和 num_gpu_blocks
    )

    # Task 4.1.2: 创建 BlockManager 和 RealModelRunner
    block_manager = BlockManager(
        # TODO
    )
    runner = RealModelRunner(
        # TODO
    )

    # Task 4.1.3: 调用 generate() 生成文本
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
