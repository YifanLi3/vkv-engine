"""
Phase 3 Test Suite — KV Cache Quantization

Run all tests:       uv run pytest tests/test_phase3.py -v
Run one Part:        uv run pytest tests/test_phase3.py -k "part1" -v
"""

import pytest
import torch

from vkv.config import CacheConfig, TINY_MODEL


# =============================================================================
# Part 1: Basic Quantization Math
# =============================================================================

class TestPart1:
    """Tests for compute_scale, quantize_tensor, dequantize_tensor."""

    def test_compute_scale_int8(self):
        from vkv.engine.quantizer import compute_scale
        x = torch.tensor([-2.0, 0.5, 1.5, -1.0])
        scale = compute_scale(x, bits=8)
        assert abs(scale.item() - 2.0 / 127) < 1e-6

    def test_compute_scale_int4(self):
        from vkv.engine.quantizer import compute_scale
        x = torch.tensor([-3.5, 1.0, 2.0])
        scale = compute_scale(x, bits=4)
        assert abs(scale.item() - 3.5 / 7) < 1e-6

    def test_compute_scale_all_positive(self):
        from vkv.engine.quantizer import compute_scale
        x = torch.tensor([1.0, 2.0, 3.0])
        scale = compute_scale(x, bits=8)
        assert abs(scale.item() - 3.0 / 127) < 1e-6

    def test_quantize_tensor_basic(self):
        from vkv.engine.quantizer import compute_scale, quantize_tensor
        x = torch.tensor([1.0, -1.0, 0.5, -0.5])
        scale = compute_scale(x, bits=8)
        q = quantize_tensor(x, scale, bits=8)
        assert q.dtype == torch.int8
        assert q.shape == x.shape
        assert q[0].item() == 127  # max value maps to 127
        assert q[1].item() == -127  # min value maps to -127

    def test_quantize_tensor_int4_clamp(self):
        from vkv.engine.quantizer import compute_scale, quantize_tensor
        x = torch.tensor([7.0, -7.0, 3.5, 0.0])
        scale = compute_scale(x, bits=4)
        q = quantize_tensor(x, scale, bits=4)
        assert q.max().item() <= 7
        assert q.min().item() >= -8

    def test_dequantize_tensor_basic(self):
        from vkv.engine.quantizer import dequantize_tensor
        q = torch.tensor([127, -127, 64, -64], dtype=torch.int8)
        scale = torch.tensor(0.01)
        x = dequantize_tensor(q, scale)
        assert x.dtype == torch.float32
        assert abs(x[0].item() - 1.27) < 1e-6

    def test_quantize_dequantize_roundtrip(self):
        from vkv.engine.quantizer import compute_scale, quantize_tensor, dequantize_tensor
        x = torch.randn(100)
        scale = compute_scale(x, bits=8)
        q = quantize_tensor(x, scale, bits=8)
        x_recon = dequantize_tensor(q, scale)
        mse = ((x - x_recon) ** 2).mean()
        assert mse < 0.001  # small error


# =============================================================================
# Part 2: Per-tensor INT8 Quantizer
# =============================================================================

class TestPart2:
    """Tests for PerTensorQuantizer."""

    def test_quantize_shape(self):
        from vkv.engine.quantizer import PerTensorQuantizer
        q = PerTensorQuantizer(bits=8)
        x = torch.randn(8, 128)
        qtensor, scale = q.quantize(x)
        assert qtensor.shape == (8, 128)
        assert qtensor.dtype == torch.int8
        assert scale.ndim == 0  # scalar

    def test_dequantize_shape(self):
        from vkv.engine.quantizer import PerTensorQuantizer
        q = PerTensorQuantizer(bits=8)
        x = torch.randn(8, 128)
        qtensor, scale = q.quantize(x)
        x_recon = q.dequantize(qtensor, scale)
        assert x_recon.shape == (8, 128)
        assert x_recon.dtype == torch.float32

    def test_roundtrip_quality(self):
        from vkv.engine.quantizer import PerTensorQuantizer
        q = PerTensorQuantizer(bits=8)
        x = torch.randn(8, 128)
        qtensor, scale = q.quantize(x)
        x_recon = q.dequantize(qtensor, scale)
        cosine = torch.nn.functional.cosine_similarity(
            x.flatten().unsqueeze(0), x_recon.flatten().unsqueeze(0)
        )
        assert cosine.item() > 0.99

    def test_3d_tensor(self):
        from vkv.engine.quantizer import PerTensorQuantizer
        q = PerTensorQuantizer(bits=8)
        x = torch.randn(8, 16, 128)  # [heads, seq_len, dim]
        qtensor, scale = q.quantize(x)
        assert qtensor.shape == (8, 16, 128)
        x_recon = q.dequantize(qtensor, scale)
        assert x_recon.shape == (8, 16, 128)


# =============================================================================
# Part 3: Per-channel INT8 Quantizer
# =============================================================================

class TestPart3:
    """Tests for PerChannelQuantizer."""

    def test_quantize_shape(self):
        from vkv.engine.quantizer import PerChannelQuantizer
        q = PerChannelQuantizer(bits=8, channel_dim=0)
        x = torch.randn(8, 128)
        qtensor, scales = q.quantize(x)
        assert qtensor.shape == (8, 128)
        assert qtensor.dtype == torch.int8
        assert scales.shape == (8,)

    def test_per_channel_independent(self):
        """Each channel should have its own scale."""
        from vkv.engine.quantizer import PerChannelQuantizer
        q = PerChannelQuantizer(bits=8, channel_dim=0)
        x = torch.zeros(4, 64)
        x[0] = torch.randn(64) * 10   # big range
        x[1] = torch.randn(64) * 0.1  # small range
        x[2] = torch.randn(64) * 5
        x[3] = torch.randn(64) * 1
        _, scales = q.quantize(x)
        assert scales[0] > scales[1]  # channel 0 has bigger scale

    def test_better_than_per_tensor(self):
        """Per-channel should be more accurate than per-tensor for varied channels."""
        from vkv.engine.quantizer import PerTensorQuantizer, PerChannelQuantizer
        x = torch.zeros(8, 128)
        x[0] = torch.randn(128) * 10  # outlier channel
        x[1:] = torch.randn(7, 128) * 0.1  # normal channels

        pt = PerTensorQuantizer(bits=8)
        pc = PerChannelQuantizer(bits=8, channel_dim=0)

        qt_pt, s_pt = pt.quantize(x)
        qt_pc, s_pc = pc.quantize(x)

        recon_pt = pt.dequantize(qt_pt, s_pt)
        recon_pc = pc.dequantize(qt_pc, s_pc)

        mse_pt = ((x - recon_pt) ** 2).mean()
        mse_pc = ((x - recon_pc) ** 2).mean()
        assert mse_pc < mse_pt  # per-channel should be better

    def test_roundtrip_quality(self):
        from vkv.engine.quantizer import PerChannelQuantizer
        q = PerChannelQuantizer(bits=8)
        x = torch.randn(8, 128)
        qtensor, scales = q.quantize(x)
        x_recon = q.dequantize(qtensor, scales)
        cosine = torch.nn.functional.cosine_similarity(
            x.flatten().unsqueeze(0), x_recon.flatten().unsqueeze(0)
        )
        assert cosine.item() > 0.99


# =============================================================================
# Part 4: Grouped INT4 Quantizer
# =============================================================================

class TestPart4:
    """Tests for GroupedQuantizer."""

    def test_quantize_shape(self):
        from vkv.engine.quantizer import GroupedQuantizer
        q = GroupedQuantizer(bits=4, group_size=32)
        x = torch.randn(8, 128)
        qtensor, scales = q.quantize(x)
        assert qtensor.shape == (8, 128)
        assert qtensor.dtype == torch.int8
        assert scales.shape == (8, 4)  # 128/32 = 4 groups per head

    def test_int4_value_range(self):
        from vkv.engine.quantizer import GroupedQuantizer
        q = GroupedQuantizer(bits=4, group_size=32)
        x = torch.randn(8, 128) * 10
        qtensor, _ = q.quantize(x)
        assert qtensor.max().item() <= 7
        assert qtensor.min().item() >= -8

    def test_roundtrip_quality(self):
        from vkv.engine.quantizer import GroupedQuantizer
        q = GroupedQuantizer(bits=4, group_size=32)
        x = torch.randn(8, 128)
        qtensor, scales = q.quantize(x)
        x_recon = q.dequantize(qtensor, scales)
        cosine = torch.nn.functional.cosine_similarity(
            x.flatten().unsqueeze(0), x_recon.flatten().unsqueeze(0)
        )
        assert cosine.item() > 0.95  # INT4 lower precision

    def test_smaller_group_better_quality(self):
        """Smaller group_size should give better precision."""
        from vkv.engine.quantizer import GroupedQuantizer
        x = torch.randn(8, 128)

        q_big = GroupedQuantizer(bits=4, group_size=128)
        q_small = GroupedQuantizer(bits=4, group_size=32)

        qt_big, s_big = q_big.quantize(x)
        qt_small, s_small = q_small.quantize(x)

        recon_big = q_big.dequantize(qt_big, s_big)
        recon_small = q_small.dequantize(qt_small, s_small)

        mse_big = ((x - recon_big) ** 2).mean()
        mse_small = ((x - recon_small) ** 2).mean()
        assert mse_small <= mse_big  # smaller group = less error


# =============================================================================
# Part 5: Quantized Cache Manager
# =============================================================================

class TestPart5:
    """Tests for QuantizedCacheManager."""

    @pytest.fixture
    def setup(self):
        from vkv.engine.block_manager import BlockManager
        from vkv.engine.quantizer import QuantizedCacheManager, PerChannelQuantizer
        mgr = BlockManager(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=50, num_cpu_blocks=20), device="cpu")
        qcm = QuantizedCacheManager(mgr, PerChannelQuantizer(bits=8))
        return mgr, qcm

    def test_write_and_read(self, setup):
        mgr, qcm = setup
        ids = mgr.allocate(1)
        key = torch.randn(4, 64, dtype=torch.float16)
        value = torch.randn(4, 64, dtype=torch.float16)

        qcm.write_quantized(ids[0], layer_idx=0, slot_idx=0, key=key.float(), value=value.float())
        k_out, v_out = qcm.read_dequantized(ids[0], layer_idx=0, slot_idx=0)

        cosine_k = torch.nn.functional.cosine_similarity(
            key.float().flatten().unsqueeze(0), k_out.flatten().unsqueeze(0)
        )
        assert cosine_k.item() > 0.99

    def test_multiple_slots(self, setup):
        mgr, qcm = setup
        ids = mgr.allocate(1)

        keys = [torch.randn(4, 64) for _ in range(3)]
        for i, k in enumerate(keys):
            qcm.write_quantized(ids[0], 0, i, k, torch.randn(4, 64))

        for i, k_orig in enumerate(keys):
            k_out, _ = qcm.read_dequantized(ids[0], 0, i)
            cosine = torch.nn.functional.cosine_similarity(
                k_orig.flatten().unsqueeze(0), k_out.flatten().unsqueeze(0)
            )
            assert cosine.item() > 0.99


# =============================================================================
# Part 6: Quantization Error Evaluation
# =============================================================================

class TestPart6:
    """Tests for compute_quantization_error."""

    def test_identical_tensors(self):
        from vkv.engine.quantizer import compute_quantization_error
        x = torch.randn(100)
        mse, cosine = compute_quantization_error(x, x)
        assert mse == 0.0
        assert abs(cosine - 1.0) < 1e-6

    def test_int8_error_is_small(self):
        from vkv.engine.quantizer import PerTensorQuantizer, compute_quantization_error
        q = PerTensorQuantizer(bits=8)
        x = torch.randn(8, 128)
        qt, s = q.quantize(x)
        x_recon = q.dequantize(qt, s)
        mse, cosine = compute_quantization_error(x, x_recon)
        assert cosine > 0.99
        assert mse < 0.01

    def test_int4_error_larger_than_int8(self):
        from vkv.engine.quantizer import PerTensorQuantizer, compute_quantization_error
        x = torch.randn(8, 128)

        q8 = PerTensorQuantizer(bits=8)
        qt8, s8 = q8.quantize(x)
        recon8 = q8.dequantize(qt8, s8)

        q4 = PerTensorQuantizer(bits=4)
        qt4, s4 = q4.quantize(x)
        recon4 = q4.dequantize(qt4, s4)

        mse8, _ = compute_quantization_error(x, recon8)
        mse4, _ = compute_quantization_error(x, recon4)
        assert mse4 > mse8  # INT4 has more error than INT8
