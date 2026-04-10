"""
Phase 2 (stub): ModelRunner — executes model forward pass.

Naming aligned with nano-vLLM/vLLM:
  nano-vLLM: ModelRunner (model_runner.py) — holds model + KV cache tensors + forward
  vLLM:      ModelRunner / GPUModelRunner — executes model on GPU workers

nano-vLLM's ModelRunner:
    - Loads model weights
    - Pre-allocates KV cache tensors
    - Runs prefill and decode forward passes
    - Applies sampling to get next token

This is a Phase 2 stub. For Phase 1 testing, use MockModelRunner below.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from vkv.config import ModelConfig


@dataclass
class MockModelConfig:
    """Configuration for the mock model runner (for Phase 1 testing)."""
    model_config: ModelConfig = None
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0


class MockModelRunner:
    """
    Mock model runner for testing Phase 1 memory management.

    Produces random KV tensors with correct shapes.
    No actual model computation.

    Usage:
        >>> runner = MockModelRunner(TINY_MODEL)
        >>> kv = runner.prefill(prompt_len=50)
        >>> # kv shape: [num_layers, 2, num_kv_heads, 50, head_dim]
    """

    def __init__(self, model_config: ModelConfig, device: str = "cpu"):
        self.config = model_config
        self.device = device

    def prefill(self, prompt_len: int) -> torch.Tensor:
        """
        Simulate prefill: return random KV cache for the entire prompt.

        Returns:
            KV tensor of shape [num_layers, 2, num_kv_heads, prompt_len, head_dim]
        """
        cfg = self.config
        return torch.randn(
            cfg.num_layers, 2, cfg.num_kv_heads, prompt_len, cfg.head_dim,
            dtype=cfg.dtype, device=self.device,
        )

    def decode_step(self) -> torch.Tensor:
        """
        Simulate one decode step: return KV for a single new token.

        Returns:
            KV tensor of shape [num_layers, 2, num_kv_heads, 1, head_dim]
        """
        cfg = self.config
        return torch.randn(
            cfg.num_layers, 2, cfg.num_kv_heads, 1, cfg.head_dim,
            dtype=cfg.dtype, device=self.device,
        )

    def sample(self, logits: Optional[torch.Tensor] = None) -> int:
        """Mock sampling: return a random token ID."""
        return torch.randint(0, 32000, (1,)).item()
