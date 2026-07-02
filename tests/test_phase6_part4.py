"""
Phase 6, Part 4 Tests: 端到端推理集成测试

Task 4.1: 单条推理
Task 4.2: 多请求并发推理
"""

import pytest
import torch

from vkv.config import ModelConfig, CacheConfig, TINY_MODEL
from vkv.engine.block_manager import BlockManager
from vkv.sampling_params import SamplingParams

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA GPU",
)


# ─────────────────────────────────────────────
# Task 4.1: 单条推理
# ─────────────────────────────────────────────

class TestPart4Task1:

    @gpu
    def test_single_generate_returns_string(self):
        """generate() 应返回非空字符串。"""
        from vkv.engine.real_model_runner import RealModelRunner

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        mgr = BlockManager(
            model_cfg,
            CacheConfig(block_size=16, num_gpu_blocks=200, num_cpu_blocks=20),
            device="cuda",
        )
        runner = RealModelRunner(MODEL_NAME, mgr, device="cuda")

        output = runner.generate("What is AI?", max_new_tokens=10)

        assert isinstance(output, str)
        assert len(output) > 0

    @gpu
    def test_single_generate_max_tokens(self):
        """生成的 token 数不超过 max_new_tokens。"""
        from vkv.engine.real_model_runner import RealModelRunner

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        mgr = BlockManager(
            model_cfg,
            CacheConfig(block_size=16, num_gpu_blocks=200, num_cpu_blocks=20),
            device="cuda",
        )
        runner = RealModelRunner(MODEL_NAME, mgr, device="cuda")

        max_new_tokens = 5
        # 每次 generate 后 block 被 free，可以复用同一个 runner
        output = runner.generate("Hello", max_new_tokens=max_new_tokens)

        generated_ids = runner.tokenizer.encode(output)
        assert len(generated_ids) <= max_new_tokens + 5  # 留一点容差

    @gpu
    def test_blocks_freed_after_generate(self):
        """generate() 结束后所有 block 应被释放。"""
        from vkv.engine.real_model_runner import RealModelRunner

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        mgr = BlockManager(
            model_cfg,
            CacheConfig(block_size=16, num_gpu_blocks=200, num_cpu_blocks=20),
            device="cuda",
        )
        runner = RealModelRunner(MODEL_NAME, mgr, device="cuda")

        runner.generate("Test prompt", max_new_tokens=5)

        # generate 内部调用 paged_cache.free()，所有 block 应归还
        assert mgr.stats.used_blocks == 0


# ─────────────────────────────────────────────
# Task 4.2: 多请求并发推理
# ─────────────────────────────────────────────

class TestPart4Task2:

    @gpu
    def test_multi_request_outputs_count(self):
        """提交 N 个请求，应返回 N 个输出。"""
        from examples.multi_inference import RealLLMEngine
        from vkv.engine.scheduler import SchedulerConfig

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=300, num_cpu_blocks=20)
        scheduler_cfg = SchedulerConfig(max_seqs_in_flight=3)

        engine = RealLLMEngine(
            model_name=MODEL_NAME,
            model_config=model_cfg,
            cache_config=cache_cfg,
            scheduler_config=scheduler_cfg,
            device="cuda",
        )
        tokenizer = engine.model_runner.tokenizer
        sp = SamplingParams(max_tokens=10)

        prompts = ["What is AI?", "Hello world.", "Explain KV cache."]
        outputs = engine.generate(
            prompts=[tokenizer.encode(p) for p in prompts],
            sampling_params=sp,
        )

        assert len(outputs) == len(prompts)

    @gpu
    def test_multi_request_no_memory_leak(self):
        """多轮推理后 block 应全部释放。"""
        from examples.multi_inference import RealLLMEngine
        from vkv.engine.scheduler import SchedulerConfig

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=300, num_cpu_blocks=20)
        scheduler_cfg = SchedulerConfig(max_seqs_in_flight=2)

        engine = RealLLMEngine(
            model_name=MODEL_NAME,
            model_config=model_cfg,
            cache_config=cache_cfg,
            scheduler_config=scheduler_cfg,
            device="cuda",
        )
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
        """每个输出的 token_ids 能被 tokenizer decode 成字符串。"""
        from examples.multi_inference import RealLLMEngine
        from vkv.engine.scheduler import SchedulerConfig

        model_cfg = ModelConfig(num_layers=22, num_kv_heads=4, head_dim=64)
        cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=300, num_cpu_blocks=20)
        scheduler_cfg = SchedulerConfig(max_seqs_in_flight=2)

        engine = RealLLMEngine(
            model_name=MODEL_NAME,
            model_config=model_cfg,
            cache_config=cache_cfg,
            scheduler_config=scheduler_cfg,
            device="cuda",
        )
        tokenizer = engine.model_runner.tokenizer
        sp = SamplingParams(max_tokens=10)

        outputs = engine.generate(
            prompts=[tokenizer.encode("What is AI?")],
            sampling_params=sp,
        )

        text = tokenizer.decode(outputs[0].output_token_ids)
        assert isinstance(text, str)
