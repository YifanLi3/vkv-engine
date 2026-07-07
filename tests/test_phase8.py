"""
Phase 8 Tests: Continuous Batching

Part 1: BatchedPagedCache
Part 2: Batched Decode
Part 3: Batched Prefill (TBD)
"""

import pytest
import torch

from vkv.config import ModelConfig, CacheConfig, TINY_MODEL
from vkv.engine.block_manager import BlockManager

has_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# ─────────────────────────────────────────────
# Part 1: BatchedPagedCache (CPU-only smoke tests)
# ─────────────────────────────────────────────

class TestPart1:

    def _make_cache(self, num_seqs: int = 0):
        from vkv.engine.batched_paged_cache import BatchedPagedCache
        model_cfg = ModelConfig(num_layers=2, num_kv_heads=4, head_dim=64)
        cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=100, num_cpu_blocks=10)
        mgr = BlockManager(model_cfg, cache_cfg, device="cpu")
        cache = BatchedPagedCache(
            block_manager=mgr,
            num_layers=model_cfg.num_layers,
            num_kv_heads=model_cfg.num_kv_heads,
            head_dim=model_cfg.head_dim,
            block_size=cache_cfg.block_size,
        )
        return mgr, cache

    def test_cpu_init_empty(self):
        _, cache = self._make_cache()
        assert cache.batch_size == 0
        assert cache.get_seq_length() == 0

    def test_cpu_add_and_remove_sequence(self):
        _, cache = self._make_cache()
        b0 = cache.add_sequence(block_table=[3, 7], seq_length=20)
        b1 = cache.add_sequence(block_table=[5], seq_length=8)
        assert cache.batch_size == 2
        assert b0 == 0 and b1 == 1
        assert cache.get_seq_length() == 20  # max

        cache.remove_sequence(0)
        assert cache.batch_size == 1
        assert cache._seq_lengths == [8]

    def test_cpu_update_writes_and_returns_padded(self):
        mgr, cache = self._make_cache()
        # 2 sequences, both starting empty; each getting 3 new tokens
        # (simulating a small batched prefill)
        cache.add_sequence(block_table=[], seq_length=0)
        cache.add_sequence(block_table=[], seq_length=0)

        B, H, T_new, D = 2, 4, 3, 64
        K = torch.randn(B, H, T_new, D)
        V = torch.randn(B, H, T_new, D)
        # Only last layer updates _seq_length
        for layer in range(2):
            full_k, full_v = cache.update(K, V, layer_idx=layer)
        # Both seqs now have _seq_length == 3
        assert cache._seq_lengths == [3, 3]
        # Output shape: [B, H, max_seq_len, D] where max_seq_len == 3
        assert full_k.shape == (B, H, 3, D)

    def test_cpu_update_different_lengths_pads(self):
        mgr, cache = self._make_cache()
        # seq 0 already has 5 tokens; seq 1 already has 2 tokens
        cache.add_sequence(block_table=mgr.allocate(1), seq_length=5)
        cache.add_sequence(block_table=mgr.allocate(1), seq_length=2)

        # Decode: 1 new token each
        B, H, D = 2, 4, 64
        K = torch.randn(B, H, 1, D)
        V = torch.randn(B, H, 1, D)
        for layer in range(2):
            full_k, full_v = cache.update(K, V, layer_idx=layer)
        # After decode: seq 0 has 6, seq 1 has 3
        assert cache._seq_lengths == [6, 3]
        # max_seq_len = 6, so output padded to 6
        assert full_k.shape == (B, H, 6, D)

    def test_cpu_free_returns_blocks(self):
        mgr, cache = self._make_cache()
        cache.add_sequence(block_table=mgr.allocate(2), seq_length=20)
        cache.add_sequence(block_table=mgr.allocate(1), seq_length=8)
        used_before = mgr.gpu_allocator.num_used
        assert used_before == 3

        cache.free()
        assert mgr.gpu_allocator.num_used == 0
        assert cache.batch_size == 0


# ─────────────────────────────────────────────
# Part 2: Batched Decode (GPU tests)
# ─────────────────────────────────────────────

class TestPart2:

    @has_gpu
    def test_gpu_batched_decode_matches_serial(self):
        """
        Batched decode should return the same tokens as serial decode
        (up to sampling randomness — use greedy / same seed).

        TODO once Part 2 implemented.
        """
        pytest.skip("Implement in Part 2")
