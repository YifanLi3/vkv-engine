"""
Phase 6, Part 4, Task 4.2: 多请求并发推理

创建 RealLLMEngine，继承 LLMEngine，替换 step() 里的 MockModelRunner
为真实 RealModelRunner，支持多请求并发。
"""

from typing import Dict, List, Optional

import torch

from vkv.config import ModelConfig, CacheConfig
from vkv.engine.block_manager import BlockManager
from vkv.engine.llm_engine import LLMEngine, RequestOutput
from vkv.engine.paged_cache import PagedCache
from vkv.engine.real_model_runner import RealModelRunner
from vkv.engine.scheduler import SchedulerConfig
from vkv.engine.sequence import SequenceStatus
from vkv.sampling_params import SamplingParams

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda"


class RealLLMEngine(LLMEngine):
    """
    LLMEngine with RealModelRunner replacing MockModelRunner.

    Key difference from LLMEngine:
    - Maintains paged_caches dict: seq_id → PagedCache
    - prefill: runner.prefill(prompt_ids) → PagedCache (KV written internally)
    - decode: runner.decode_step(last_token, paged_cache) → (logits, paged_cache)
    - on finish: paged_cache.free()
    """

    def __init__(
        self,
        model_name: str,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig = None,
        device: str = "cuda",
    ):
        super().__init__(model_config, cache_config, scheduler_config, device)

        # Task 4.2.1: 替换 MockModelRunner 为 RealModelRunner
        self.model_runner = RealModelRunner(
            # TODO: 传入 model_name, block_manager, device
        )

        # Task 4.2.2: 维护每个 seq 的 PagedCache
        self.paged_caches: Dict[int, PagedCache] = {}

    def step(self) -> List[RequestOutput]:
        """
        Override step() to use real model inference.

        Prefill step:
            1. scheduler.schedule() → SchedulerOutput (is_prefill=True)
            2. For each seq:
               a. 调用 self.model_runner.prefill(seq 的 prompt token ids)
               b. 把返回的 PagedCache 存进 self.paged_caches[seq.seq_id]
            3. Return []  (prefill 不产生新 token)

        Decode step:
            1. scheduler.schedule() → SchedulerOutput (is_prefill=False)
            2. For each seq:
               a. 取出 self.paged_caches[seq.seq_id]
               b. 调用 self.model_runner.decode_step(last_token_id, paged_cache)
               c. 调用 self.model_runner.sample(logits) 得到 token_id
            3. scheduler.postprocess(seqs, token_ids)
            4. 对于完成的 seq：paged_cache.free()，收集 RequestOutput
            5. Return finished outputs
        """
        output = self.scheduler.schedule()

        if output.is_prefill:
            for seq in output.scheduled_seqs:
                # TODO: prefill 每个 seq，把 PagedCache 存入 self.paged_caches
                pass
            return []
        else:
            token_ids = []
            for seq in output.scheduled_seqs:
                # TODO: decode 每个 seq
                # 提示：seq.token_ids[-1] 是上一个 token
                pass

            finished_seqs = self.scheduler.postprocess(output.scheduled_seqs, token_ids)
            for seq in finished_seqs:
                # TODO: free PagedCache，收集 RequestOutput
                pass

            return [self.outputs[seq.seq_id] for seq in finished_seqs]


def main():
    # Task 4.2.3: 初始化配置
    model_cfg = ModelConfig(
        # TODO: TinyLlama 参数
    )
    cache_cfg = CacheConfig(
        # TODO
    )
    scheduler_cfg = SchedulerConfig(
        # TODO: 设置 max_seqs_in_flight 等参数
    )

    engine = RealLLMEngine(
        model_name=MODEL_NAME,
        model_config=model_cfg,
        cache_config=cache_cfg,
        scheduler_config=scheduler_cfg,
        device=DEVICE,
    )

    tokenizer = engine.model_runner.tokenizer

    # Task 4.2.4: 提交多个请求
    prompts = [
        "What is AI?",
        "Explain neural networks.",
        "What is KV cache?",
    ]

    sp = SamplingParams(max_tokens=30)
    outputs = engine.generate(
        prompts=[tokenizer.encode(p) for p in prompts],
        sampling_params=sp,
    )

    for prompt, out in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Output: {tokenizer.decode(out.output_token_ids)}")


if __name__ == "__main__":
    main()
