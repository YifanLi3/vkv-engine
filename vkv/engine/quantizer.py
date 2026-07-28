"""
Phase 3: KV Cache Quantization

Reduces KV cache memory by 50% (INT8) or 75% (INT4) with minimal precision loss.

No direct nano-vLLM equivalent — nano-vLLM doesn't support quantized KV cache.
vLLM has FP8 KV cache support. Our implementation covers INT8 and INT4.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


# =============================================================================
# Part 1: Basic Quantization Math
# =============================================================================

def compute_scale(tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Compute the quantization scale for a tensor.

    scale = max(|x|) / (2^(bits-1) - 1)

    For INT8: scale = max(|x|) / 127
    For INT4: scale = max(|x|) / 7

    Args:
        tensor: Input FP16/FP32 tensor
        bits: Target bit width (8 or 4)

    Returns:
        Scalar scale value

    Example:
        >>> x = torch.tensor([-2.0, 0.5, 1.5, -1.0])
        >>> compute_scale(x, bits=8)
        tensor(0.01575)  # 2.0 / 127

    """
    max_val = 2 ** (bits - 1) - 1
    return tensor.abs().max() / max_val


def quantize_tensor(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    bits: int = 8,
) -> torch.Tensor:
    """
    Quantize a FP tensor to integer using the given scale.

    q = clamp(round(x / scale), min_val, max_val)

    For INT8: clamp to [-128, 127]
    For INT4: clamp to [-8, 7]

    Args:
        tensor: Input FP tensor
        scale: Quantization scale
        bits: Target bit width

    Returns:
        Quantized tensor (dtype=torch.int8)

    Note: Even for INT4, we store in int8 dtype (PyTorch has no int4).
    """
    min_val = - 2 ** (bits - 1)
    max_val = 2 ** (bits - 1) - 1
    return torch.clamp(torch.round(tensor/scale), min_val, max_val).to(torch.int8)
    

def dequantize_tensor(
    qtensor: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Dequantize an integer tensor back to FP.

    x_approx = q * scale

    Args:
        qtensor: Quantized int8 tensor
        scale: Quantization scale (same as used in quantize)

    Returns:
        Dequantized FP32 tensor

    """
    return qtensor.to(torch.float32) * scale


# =============================================================================
# Part 2: Per-tensor INT8 Quantizer
# =============================================================================

class PerTensorQuantizer:
    """
    Simplest quantizer: one scale for the entire tensor.

    Good: minimal storage overhead (1 float per tensor)
    Bad: if one value is an outlier, all values lose precision

    Usage:
        >>> q = PerTensorQuantizer(bits=8)
        >>> qtensor, scale = q.quantize(key_tensor)
        >>> key_approx = q.dequantize(qtensor, scale)
    """

    def __init__(self, bits: int = 8):
        self.bits = bits

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize entire tensor with a single scale.

        Args:
            tensor: FP tensor of any shape, e.g. [num_kv_heads, head_dim]

        Returns:
            (quantized_tensor, scale)
            quantized_tensor: same shape, dtype=int8
            scale: scalar tensor

        1. Compute scale using compute_scale()
        2. Quantize using quantize_tensor()
        3. Return (quantized, scale)
        """
        scale = compute_scale(tensor, self.bits)
        qtensor = quantize_tensor(tensor, scale, self.bits)
        return qtensor, scale

    def dequantize(self, qtensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Dequantize back to FP.

        Use dequantize_tensor().
        """
        return dequantize_tensor(qtensor, scale)


# =============================================================================
# Part 3: Per-channel INT8 Quantizer
# =============================================================================

class PerChannelQuantizer:
    """
    One scale per KV head — better precision than per-tensor.

    For tensor shape [num_kv_heads, seq_len, head_dim]:
      scales shape: [num_kv_heads] — one scale per head

    Each head is quantized independently, so different heads
    can have very different value ranges without hurting each other.
    """

    def __init__(self, bits: int = 8, channel_dim: int = 0):
        """
        Args:
            bits: Target bit width
            channel_dim: Which dimension is the "channel" (head) dimension.
                         Default 0 for shape [num_kv_heads, head_dim].
        """
        self.bits = bits
        self.channel_dim = channel_dim

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize with one scale per channel.

        For tensor [num_kv_heads, head_dim] with channel_dim=0:
          scales[i] = max(|tensor[i, :]|) / 127

        Args:
            tensor: FP tensor, e.g. [num_kv_heads, head_dim]

        Returns:
            (quantized_tensor, scales)
            scales shape: [num_kv_heads]

        Hint: Use tensor.abs().amax(dim=...) to get per-channel max.
        """
        max_val = 2 ** (self.bits - 1) - 1

        # 1. For each channel (head), compute its own scale
        dims = [d for d in range(tensor.ndim) if d != self.channel_dim]
        scales = tensor.abs().amax(dim=dims) / max_val

        # 2. Reshape scale for broadcasting
        shape = [1] * tensor.ndim
        shape[self.channel_dim] = -1
        reshaped_scales = scales.reshape(shape)

        # 3. Quantize: round(tensor / scale_broadcast)
        qtensor = torch.clamp(
                torch.round(tensor/reshaped_scales), -max_val - 1, max_val
        ).to(torch.int8)

        # 4. Return (quantized, scales)
        return qtensor, scales

    def dequantize(self, qtensor: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        Dequantize with per-channel scales.

        Reshape scales for broadcasting, then qtensor.float() * scales.
        """
        shape = [1] * qtensor.ndim
        shape[self.channel_dim] = -1
        reshaped_scales = scales.reshape(shape)

        return qtensor.to(torch.float32) * reshaped_scales


# =============================================================================
# Part 4: Grouped INT4 Quantizer
# =============================================================================

class GroupedQuantizer:
    """
    Grouped quantization — one scale per group of elements.

    For INT4 (only 16 values), per-tensor scale is too coarse.
    Grouping gives much better precision.

    For tensor [num_kv_heads, head_dim] with group_size=32:
      head_dim=128 → 4 groups per head → 4 scales per head
      scales shape: [num_kv_heads, num_groups]

    Trade-off: more scales = more storage overhead, but better precision.
    """

    def __init__(self, bits: int = 4, group_size: int = 32):
        self.bits = bits
        self.group_size = group_size

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize with one scale per group.

        Args:
            tensor: FP tensor [num_kv_heads, head_dim]

        Returns:
            (quantized_tensor, scales)
            quantized_tensor: same shape [num_kv_heads, head_dim], dtype=int8
            scales: [num_kv_heads, num_groups]

        """
        max_val = 2 ** (self.bits - 1) - 1
        num_kv_heads, head_dim = tensor.shape
        num_groups = head_dim // self.group_size

        # 1. Reshape to [num_kv_heads, num_groups, group_size]
        reshaped_tensor = tensor.view(num_kv_heads, num_groups, self.group_size)

        # 2. Compute scale per group: [num_kv_heads, num_groups]
        scales = reshaped_tensor.abs().amax(dim=-1) / max_val

        # 3. Quantize each group
        qtensor = torch.clamp(
                torch.round(reshaped_tensor/scales.unsqueeze(-1)), -max_val - 1, max_val
        ).to(torch.int8)

        # 4. Reshape back to [num_kv_heads, head_dim]
        qtensor = qtensor.view(num_kv_heads, head_dim)

        return qtensor, scales

    def dequantize(self, qtensor: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        Dequantize with per-group scales.

        """
        # 1. Reshape qtensor to [num_kv_heads, num_groups, group_size]
        num_kv_heads, head_dim = qtensor.shape
        num_groups = head_dim // self.group_size
        qtensor = qtensor.view(num_kv_heads, num_groups, self.group_size)

        # 2. Multiply by scales.unsqueeze(-1)
        result = qtensor.to(torch.float32) * scales.unsqueeze(-1)

        # 3. Reshape back
        return result.view(num_kv_heads, head_dim)

# =============================================================================
# Part 5: Quantized Cache Manager
# =============================================================================

class QuantizedCacheManager:
    """
    Wraps BlockManager to support quantized KV cache storage.

    Normal BlockManager: stores FP16 KV → 2 bytes per element
    QuantizedCacheManager: stores INT8 KV → 1 byte per element (50% savings)
                           stores INT4 KV → 0.5 byte per element (75% savings)

    The scale factors are stored separately for each (block, layer, slot).
    """

    def __init__(self, block_manager, quantizer=None):
        """
        Args:
            block_manager: BlockManager from Phase 1
            quantizer: Quantizer instance (PerTensor, PerChannel, or Grouped)
                       If None, defaults to PerChannelQuantizer(bits=8)

        TODO: Implement this.
        1. Store block_manager reference
        2. Store quantizer (default to PerChannelQuantizer)
        3. Initialize scale storage: Dict keyed by (block_id, layer_idx, slot_idx)
        """
        raise NotImplementedError("TODO: Implement QuantizedCacheManager.__init__")

    def write_quantized(
        self,
        block_id: int,
        layer_idx: int,
        slot_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Quantize KV data and write to BlockManager.

        Args:
            block_id, layer_idx, slot_idx: Where to write
            key: FP16 key tensor [num_kv_heads, head_dim]
            value: FP16 value tensor [num_kv_heads, head_dim]

        TODO: Implement this.
        1. Quantize key → (q_key, k_scale)
        2. Quantize value → (q_value, v_scale)
        3. Write q_key and q_value to block_manager (as int8 tensors)
        4. Store scales in self._scales[(block_id, layer_idx, slot_idx)]
        """
        raise NotImplementedError("TODO: Implement write_quantized")

    def read_dequantized(
        self,
        block_id: int,
        layer_idx: int,
        slot_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read quantized KV data from BlockManager and dequantize.

        Returns:
            (key, value) as FP tensors (approximately restored)

        TODO: Implement this.
        1. Read q_key, q_value from block_manager
        2. Retrieve scales from self._scales
        3. Dequantize both
        4. Return (key_approx, value_approx)
        """
        raise NotImplementedError("TODO: Implement read_dequantized")


# =============================================================================
# Part 6: Quantization Error Evaluation
# =============================================================================

def compute_quantization_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> Tuple[float, float]:
    """
    Measure quantization error between original and reconstructed tensors.

    Args:
        original: Original FP tensor
        reconstructed: Dequantized (approximately restored) tensor

    Returns:
        (mse, cosine_similarity)
        mse: Mean Squared Error (lower = better)
        cosine_similarity: Cosine similarity (closer to 1.0 = better)

    TODO: Implement this.
    mse = ((original - reconstructed) ** 2).mean().item()
    cosine_sim = F.cosine_similarity(
        original.flatten().unsqueeze(0),
        reconstructed.flatten().unsqueeze(0)
    ).item()
    """
    raise NotImplementedError("TODO: Implement compute_quantization_error")
