"""
Global configuration for vkv-engine.
These are provided for you — no TODOs here.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """Model architecture parameters that affect KV cache shape."""
    num_layers: int = 32
    num_kv_heads: int = 8
    num_attention_heads: int = 32
    head_dim: int = 128
    dtype: torch.dtype = torch.float16

    @property
    def kv_head_group_size(self) -> int:
        """For GQA: how many Q heads share one KV head."""
        return self.num_attention_heads // self.num_kv_heads

    @property
    def element_size(self) -> int:
        """Bytes per element for the configured dtype."""
        return torch.tensor([], dtype=self.dtype).element_size()


@dataclass
class CacheConfig:
    """Configuration for the KV cache memory pool."""
    block_size: int = 16
    num_gpu_blocks: Optional[int] = None
    num_cpu_blocks: int = 256
    gpu_memory_fraction: float = 0.90
    swap_space_gb: float = 4.0
    eviction_policy: str = "lru"

    def __post_init__(self):
        assert self.block_size > 0 and (self.block_size & (self.block_size - 1)) == 0, \
            f"block_size must be a power of 2, got {self.block_size}"
        assert 0 < self.gpu_memory_fraction <= 1.0


# Pre-built configs for common models (useful for testing)
LLAMA_8B = ModelConfig(num_layers=32, num_kv_heads=8, num_attention_heads=32, head_dim=128)
LLAMA_70B = ModelConfig(num_layers=80, num_kv_heads=8, num_attention_heads=64, head_dim=128)
TINY_MODEL = ModelConfig(num_layers=4, num_kv_heads=4, num_attention_heads=4, head_dim=64)
