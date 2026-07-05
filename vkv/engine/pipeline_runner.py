"""
Phase 7, Part 2: Pipeline Parallelism Runner

Splits transformer layers across multiple GPUs. A single request flows
through all GPUs sequentially:
    GPU 0: layers 0-10
    GPU 1: layers 11-21

Uses HuggingFace `accelerate`'s device_map for layer sharding.
Activations are moved between GPUs automatically by HF hooks.

KV cache follows the same partitioning: layer L's KV lives on the same
GPU as layer L's weights. This is achieved by passing `layer_device_map`
to BlockManager.
"""

from typing import Dict, List, Optional, Tuple

import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from vkv.config import ModelConfig
from vkv.engine.block_manager import BlockManager
from vkv.engine.paged_cache import PagedCache
from vkv.engine.real_model_runner import RealModelRunner


class PipelineParallelRunner(RealModelRunner):
    """
    RealModelRunner variant that splits model layers across N GPUs.

    Assumes:
      - Model has N * layers_per_gpu transformer layers (approximate split OK)
      - Embedding / lm_head can go on GPU 0 and GPU N-1 respectively

    Usage:
        block_manager = BlockManager(model_config, cache_config,
                                     layer_device_map=layer_device_map)
        runner = PipelineParallelRunner(
            model_name="...", block_manager=block_manager,
            num_gpus=2,
        )
    """

    def __init__(
        self,
        model_name: str,
        block_manager: BlockManager,
        num_gpus: int = 2,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Load model with pipeline-parallel device_map.

        Steps:
        1. Peek at model config to know num_layers (before loading full weights)
        2. Compute device_map: which layer -> which GPU
        3. Load model with HF's device_map
        4. Store references (block_manager, block_size, model_config, ...)

        Note: unlike RealModelRunner, we CANNOT reuse its __init__ directly,
        because that calls `.to(device)` which conflicts with device_map.
        """
        # We deliberately do NOT call super().__init__(), to avoid its .to(device) logic.

        cfg = AutoConfig.from_pretrained(model_name)
        num_layers = cfg.num_hidden_layers

        # Build device_map: split layers evenly across `num_gpus`
        self.device_map = self._build_device_map(num_layers, num_gpus)
        # For convenience: layer_idx -> device string (e.g. "cuda:0")
        self.layer_device_map = {
            i: f"cuda:{self.device_map[f'model.layers.{i}']}"
            for i in range(num_layers)
        }

        # Sanity check: block_manager's layer_device_map should match ours
        if block_manager.layer_device_map != self.layer_device_map:
            raise ValueError(
                "block_manager.layer_device_map does not match this runner's "
                "device split. Create BlockManager with the same layer_device_map."
            )

        self.dtype = dtype
        self.block_manager = block_manager
        self.block_size = block_manager.block_size
        # For "device" attribute compatibility with parent class; use GPU 0 as reference
        self.device = "cuda:0"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device_map,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_config = self._extract_model_config()

    @staticmethod
    def _build_device_map(num_layers: int, num_gpus: int) -> Dict[str, int]:
        """
        Compute the HuggingFace device_map dict.

        Layout:
          - Embedding on GPU 0
          - Layers split evenly (round-up so last GPU may take fewer)
          - final norm + lm_head on GPU (num_gpus - 1)
        """
        layers_per_gpu = (num_layers + num_gpus - 1) // num_gpus
        device_map = {
            "model.embed_tokens": 0,
            "model.norm": num_gpus - 1,
            "lm_head": num_gpus - 1,
        }
        for layer_idx in range(num_layers):
            device_map[f"model.layers.{layer_idx}"] = min(
                layer_idx // layers_per_gpu, num_gpus - 1
            )
        return device_map
