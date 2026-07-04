"""
Phase 6 Test Suite — Real Model Integration

Tests marked with @pytest.mark.gpu require a GPU.
CPU-only tests use mock/small tensors.

Run CPU tests:   uv run pytest tests/test_phase6.py -k "cpu" -v
Run GPU tests:   uv run pytest tests/test_phase6.py -k "gpu" -v
Run all:         uv run pytest tests/test_phase6.py -v
"""

import pytest
import torch

from vkv.config import ModelConfig, CacheConfig, TINY_MODEL


HAS_CUDA = torch.cuda.is_available()
gpu = pytest.mark.skipif(not HAS_CUDA, reason="Requires CUDA GPU")


# =============================================================================
# Part 1: Model Loading (CPU tests with TinyLlama or mock)
# =============================================================================

class TestPart1:
    """Tests for model loading and config extraction."""

    def test_cpu_extract_model_config(self):
        """Test config extraction without loading a real model."""
        config = ModelConfig(num_layers=4, num_kv_heads=4, head_dim=64)
        assert config.num_layers == 4
        assert config.num_kv_heads == 4

    @gpu
    def test_gpu_load_tiny_model(self):
        """Load a small model on GPU."""
        from vkv.engine.real_model_runner import RealModelRunner
        mgr_cfg = CacheConfig(block_size=16, num_gpu_blocks=100, num_cpu_blocks=20)
        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        mgr = __import__('vkv.engine.block_manager', fromlist=['BlockManager']).BlockManager(
            model_cfg, mgr_cfg, device="cuda"
        )
        runner = RealModelRunner(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            block_manager=mgr,
            device="cuda",
        )
        assert runner.model is not None
        assert runner.tokenizer is not None


# =============================================================================
# Part 2: PagedCache (CPU tests with manual tensors)
# =============================================================================

class TestPart2:
    """Tests for PagedCache — can run on CPU with mock data."""

    @pytest.fixture
    def setup(self):
        from vkv.engine.block_manager import BlockManager
        from vkv.engine.paged_cache import PagedCache
        mgr = BlockManager(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=50, num_cpu_blocks=20), device="cpu")
        cache = PagedCache(
            block_manager=mgr,
            num_layers=TINY_MODEL.num_layers,
            num_kv_heads=TINY_MODEL.num_kv_heads,
            head_dim=TINY_MODEL.head_dim,
            block_size=16,
        )
        return mgr, cache

    def test_cpu_paged_cache_init(self, setup):
        mgr, cache = setup
        assert cache.get_seq_length() == 0
        assert len(cache.block_table) == 0

    def test_cpu_paged_cache_update_single_token(self, setup):
        mgr, cache = setup
        key = torch.randn(1, TINY_MODEL.num_kv_heads, 1, TINY_MODEL.head_dim)
        val = torch.randn(1, TINY_MODEL.num_kv_heads, 1, TINY_MODEL.head_dim)

        for layer_idx in range(TINY_MODEL.num_layers):
            full_k, full_v = cache.update(key, val, layer_idx)

        assert cache.get_seq_length() == 1
        assert len(cache.block_table) == 1

    def test_cpu_paged_cache_update_multiple_tokens(self, setup):
        mgr, cache = setup

        for token_idx in range(5):
            key = torch.randn(1, TINY_MODEL.num_kv_heads, 1, TINY_MODEL.head_dim)
            val = torch.randn(1, TINY_MODEL.num_kv_heads, 1, TINY_MODEL.head_dim)
            for layer_idx in range(TINY_MODEL.num_layers):
                full_k, full_v = cache.update(key, val, layer_idx)

        assert cache.get_seq_length() == 5

    def test_cpu_paged_cache_update_returns_full_kv(self, setup):
        mgr, cache = setup

        for token_idx in range(3):
            key = torch.randn(1, TINY_MODEL.num_kv_heads, 1, TINY_MODEL.head_dim)
            val = torch.randn(1, TINY_MODEL.num_kv_heads, 1, TINY_MODEL.head_dim)
            for layer_idx in range(TINY_MODEL.num_layers):
                full_k, full_v = cache.update(key, val, layer_idx)

        assert full_k.shape == (1, TINY_MODEL.num_kv_heads, 3, TINY_MODEL.head_dim)
        assert full_v.shape == (1, TINY_MODEL.num_kv_heads, 3, TINY_MODEL.head_dim)

    def test_cpu_paged_cache_block_allocation(self, setup):
        mgr, cache = setup

        # Fill exactly one block (16 tokens)
        for token_idx in range(16):
            key = torch.randn(1, TINY_MODEL.num_kv_heads, 1, TINY_MODEL.head_dim)
            val = torch.randn(1, TINY_MODEL.num_kv_heads, 1, TINY_MODEL.head_dim)
            for layer_idx in range(TINY_MODEL.num_layers):
                cache.update(key, val, layer_idx)

        assert len(cache.block_table) == 1  # exactly 1 block

        # Add one more token → should allocate second block
        key = torch.randn(1, TINY_MODEL.num_kv_heads, 1, TINY_MODEL.head_dim)
        val = torch.randn(1, TINY_MODEL.num_kv_heads, 1, TINY_MODEL.head_dim)
        for layer_idx in range(TINY_MODEL.num_layers):
            cache.update(key, val, layer_idx)

        assert len(cache.block_table) == 2  # new block allocated

    def test_cpu_paged_cache_free(self, setup):
        mgr, cache = setup

        for token_idx in range(5):
            key = torch.randn(1, TINY_MODEL.num_kv_heads, 1, TINY_MODEL.head_dim)
            val = torch.randn(1, TINY_MODEL.num_kv_heads, 1, TINY_MODEL.head_dim)
            for layer_idx in range(TINY_MODEL.num_layers):
                cache.update(key, val, layer_idx)

        assert mgr.stats.used_blocks > 0
        cache.free()
        assert mgr.stats.used_blocks == 0
        assert cache.get_seq_length() == 0

    def test_cpu_paged_cache_prefill_batch(self, setup):
        """Simulate prefill: multiple tokens at once."""
        mgr, cache = setup
        prompt_len = 10

        key = torch.randn(1, TINY_MODEL.num_kv_heads, prompt_len, TINY_MODEL.head_dim)
        val = torch.randn(1, TINY_MODEL.num_kv_heads, prompt_len, TINY_MODEL.head_dim)

        for layer_idx in range(TINY_MODEL.num_layers):
            full_k, full_v = cache.update(key, val, layer_idx)

        assert cache.get_seq_length() == prompt_len
        assert full_k.shape[2] == prompt_len


# =============================================================================
# Part 3: RealModelRunner (GPU tests)
# =============================================================================

class TestPart3:
    """Tests for RealModelRunner — requires GPU."""

    @gpu
    def test_gpu_prefill(self):
        from vkv.engine.real_model_runner import RealModelRunner
        from vkv.engine.block_manager import BlockManager

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        mgr = BlockManager(model_cfg, CacheConfig(block_size=16, num_gpu_blocks=200, num_cpu_blocks=20), device="cuda")
        runner = RealModelRunner("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mgr, device="cuda")

        cache = runner.prefill([1, 2, 3, 4, 5])
        assert cache.get_seq_length() == 5

    @gpu
    def test_gpu_decode_step(self):
        from vkv.engine.real_model_runner import RealModelRunner
        from vkv.engine.block_manager import BlockManager

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        mgr = BlockManager(model_cfg, CacheConfig(block_size=16, num_gpu_blocks=200, num_cpu_blocks=20), device="cuda")
        runner = RealModelRunner("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mgr, device="cuda")

        cache = runner.prefill([1, 2, 3, 4, 5])
        logits, cache = runner.decode_step(token_id=100, paged_cache=cache)
        assert logits.shape[-1] == runner.model.config.vocab_size
        assert cache.get_seq_length() == 6

    @gpu
    def test_gpu_generate(self):
        from vkv.engine.real_model_runner import RealModelRunner
        from vkv.engine.block_manager import BlockManager

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        mgr = BlockManager(model_cfg, CacheConfig(block_size=16, num_gpu_blocks=200, num_cpu_blocks=20), device="cuda")
        runner = RealModelRunner("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mgr, device="cuda")

        output = runner.generate("What is AI?", max_new_tokens=20)
        assert isinstance(output, str)
        assert len(output) > 0
        assert mgr.stats.used_blocks == 0  # cache freed after generate

    @gpu
    def test_gpu_no_memory_leak(self):
        from vkv.engine.real_model_runner import RealModelRunner
        from vkv.engine.block_manager import BlockManager

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        mgr = BlockManager(model_cfg, CacheConfig(block_size=16, num_gpu_blocks=200, num_cpu_blocks=20), device="cuda")
        runner = RealModelRunner("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mgr, device="cuda")

        for _ in range(3):
            runner.generate("Hello", max_new_tokens=10)

        assert mgr.stats.used_blocks == 0


# =============================================================================
# Part 4: End-to-end Inference
# =============================================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class TestPart4Task1:

    @gpu
    def test_single_generate_returns_string(self):
        """generate() should return a non-empty string."""
        from vkv.engine.real_model_runner import RealModelRunner
        from vkv.engine.block_manager import BlockManager

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        mgr = BlockManager(model_cfg, CacheConfig(block_size=16, num_gpu_blocks=200, num_cpu_blocks=20), device="cuda")
        runner = RealModelRunner(MODEL_NAME, mgr, device="cuda")

        output = runner.generate("What is AI?", max_new_tokens=10)

        assert isinstance(output, str)
        assert len(output) > 0

    @gpu
    def test_single_generate_max_tokens(self):
        """Number of generated tokens should not exceed max_new_tokens."""
        from vkv.engine.real_model_runner import RealModelRunner
        from vkv.engine.block_manager import BlockManager

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        mgr = BlockManager(model_cfg, CacheConfig(block_size=16, num_gpu_blocks=200, num_cpu_blocks=20), device="cuda")
        runner = RealModelRunner(MODEL_NAME, mgr, device="cuda")

        max_new_tokens = 5
        output = runner.generate("Hello", max_new_tokens=max_new_tokens)

        generated_ids = runner.tokenizer.encode(output)
        assert len(generated_ids) <= max_new_tokens + 5

    @gpu
    def test_blocks_freed_after_generate(self):
        """All blocks should be released after generate() completes."""
        from vkv.engine.real_model_runner import RealModelRunner
        from vkv.engine.block_manager import BlockManager

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        mgr = BlockManager(model_cfg, CacheConfig(block_size=16, num_gpu_blocks=200, num_cpu_blocks=20), device="cuda")
        runner = RealModelRunner(MODEL_NAME, mgr, device="cuda")

        runner.generate("Test prompt", max_new_tokens=5)

        assert mgr.stats.used_blocks == 0


class TestPart4Task2:

    @gpu
    def test_multi_request_outputs_count(self):
        """Submitting N requests should return N outputs."""
        from vkv.engine.real_llm_engine import RealLLMEngine
        from vkv.engine.scheduler import SchedulerConfig
        from vkv.sampling_params import SamplingParams

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=300, num_cpu_blocks=20)
        scheduler_cfg = SchedulerConfig(max_num_seqs=3)

        engine = RealLLMEngine(MODEL_NAME, model_cfg, cache_cfg, scheduler_cfg, device="cuda")
        tokenizer = engine.model_runner.tokenizer
        sp = SamplingParams(max_tokens=10)

        prompts = ["What is AI?", "Hello world.", "Explain KV cache."]
        outputs = engine.generate(prompts=[tokenizer.encode(p) for p in prompts], sampling_params=sp)

        assert len(outputs) == len(prompts)

    @gpu
    def test_multi_request_no_memory_leak(self):
        """All blocks should be freed after each generate() call."""
        from vkv.engine.real_llm_engine import RealLLMEngine
        from vkv.engine.scheduler import SchedulerConfig
        from vkv.sampling_params import SamplingParams

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=300, num_cpu_blocks=20)
        scheduler_cfg = SchedulerConfig(max_num_seqs=2)

        engine = RealLLMEngine(MODEL_NAME, model_cfg, cache_cfg, scheduler_cfg, device="cuda")
        tokenizer = engine.model_runner.tokenizer
        sp = SamplingParams(max_tokens=5)

        for _ in range(3):
            engine.generate(
                prompts=[tokenizer.encode("Hello"), tokenizer.encode("Hi there")],
                sampling_params=sp,
            )
            assert engine.block_manager.stats.used_blocks == 0

    @gpu
    def test_multi_request_output_is_decodable(self):
        """Each output's token_ids should be decodable to a string."""
        from vkv.engine.real_llm_engine import RealLLMEngine
        from vkv.engine.scheduler import SchedulerConfig
        from vkv.sampling_params import SamplingParams

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=300, num_cpu_blocks=20)
        scheduler_cfg = SchedulerConfig(max_num_seqs=2)

        engine = RealLLMEngine(MODEL_NAME, model_cfg, cache_cfg, scheduler_cfg, device="cuda")
        tokenizer = engine.model_runner.tokenizer
        sp = SamplingParams(max_tokens=10)

        outputs = engine.generate(prompts=[tokenizer.encode("What is AI?")], sampling_params=sp)

        text = tokenizer.decode(outputs[0].output_token_ids)
        assert isinstance(text, str)
