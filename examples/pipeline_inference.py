"""
Phase 7, Part 2: Pipeline Parallel Inference Demo

Runs a single-process inference with model layers split across N GPUs.

Requires >= 2 GPUs. HF's device_map handles activation transfer between
GPUs automatically via forward-pre hooks.

Run:
    uv run python examples/pipeline_inference.py
"""

from transformers import AutoConfig

from vkv.config import ModelConfig, CacheConfig
from vkv.engine.block_manager import BlockManager
from vkv.engine.pipeline_runner import PipelineParallelRunner

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NUM_GPUS = 2
MAX_NEW_TOKENS = 30


def main():
    import torch
    assert torch.cuda.device_count() >= NUM_GPUS, \
        f"Need at least {NUM_GPUS} GPUs, found {torch.cuda.device_count()}"

    # ─── Task 2.3.1: derive ModelConfig from HF config ───
    # TODO: peek AutoConfig, build ModelConfig with num_layers/num_kv_heads/head_dim
    hf_cfg = AutoConfig.from_pretrained(MODEL_NAME)
    model_cfg = ModelConfig(
        num_layers=hf_cfg.num_hidden_layers,
        num_kv_heads=hf_cfg.num_key_value_heads,
        head_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
    )
    cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=300, num_cpu_blocks=20)

    # ─── Task 2.3.2: build layer -> device map ───
    # PipelineParallelRunner._build_device_map returns "model.layers.i" -> gpu_id
    # We need layer_idx -> "cuda:N" for BlockManager
    hf_device_map = PipelineParallelRunner._build_device_map(
        model_cfg.num_layers, NUM_GPUS
    )
    layer_device_map = {
        i: f"cuda:{hf_device_map[f'model.layers.{i}']}"
        for i in range(model_cfg.num_layers)
    }
    print("Layer -> device:")
    for i, d in layer_device_map.items():
        print(f"  layer {i:2d} -> {d}")

    # ─── Task 2.3.3: create BlockManager with per-layer devices ───
    block_manager = BlockManager(
        model_cfg, cache_cfg,
        device="cuda:0",  # fallback; ignored per-layer since layer_device_map is set
        layer_device_map=layer_device_map,
    )

    # ─── Task 2.3.4: create runner and generate ───
    runner = PipelineParallelRunner(
        model_name=MODEL_NAME,
        block_manager=block_manager,
        num_gpus=NUM_GPUS,
    )

    prompts = [
        "What is artificial intelligence?",
        "Explain neural networks briefly.",
    ]
    print("\n" + "=" * 60)
    print("Pipeline Parallel Inference Results")
    print("=" * 60)
    for prompt in prompts:
        output = runner.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
        print(f"\nPrompt: {prompt}\nOutput: {output}")


if __name__ == "__main__":
    main()
