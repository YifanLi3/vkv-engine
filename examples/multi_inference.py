"""
Phase 6, Part 4, Task 4.2: Multi-request concurrent inference

Create RealLLMEngine by extending LLMEngine and overriding step() to use
RealModelRunner instead of MockModelRunner. Supports multiple concurrent requests.
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

    Key differences from LLMEngine:
    - Maintains paged_caches dict: seq_id -> PagedCache
    - prefill: runner.prefill(prompt_ids) -> PagedCache (KV written internally)
    - decode: runner.decode_step(last_token, paged_cache) -> (logits, paged_cache)
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

        # Task 4.2.1: Replace MockModelRunner with RealModelRunner
        self.model_runner = RealModelRunner(
            # TODO: pass model_name, block_manager, device
        )

        # Maintains one PagedCache per active sequence
        self.paged_caches: Dict[int, PagedCache] = {}

    def step(self) -> List[RequestOutput]:
        """
        Override step() to use real model inference.

        Prefill step:
            1. scheduler.schedule() -> SchedulerOutput (is_prefill=True)
            2. For each seq:
               a. Call self.model_runner.prefill(seq's prompt token ids)
               b. Store the returned PagedCache in self.paged_caches[seq.seq_id]
            3. Return []  (no tokens generated during prefill)

        Decode step:
            1. scheduler.schedule() -> SchedulerOutput (is_prefill=False)
            2. For each seq:
               a. Retrieve self.paged_caches[seq.seq_id]
               b. Call self.model_runner.decode_step(last_token_id, paged_cache)
               c. Call self.model_runner.sample(logits) to get token_id
            3. scheduler.postprocess(seqs, token_ids)
            4. For finished seqs: paged_cache.free(), collect RequestOutput
            5. Return finished outputs
        """
        output = self.scheduler.schedule()

        if output.is_prefill:
            for seq in output.scheduled_seqs:
                # TODO: prefill each seq, store PagedCache in self.paged_caches
                pass
            return []
        else:
            token_ids = []
            for seq in output.scheduled_seqs:
                # TODO: decode each seq
                # Hint: seq.token_ids[-1] is the last generated token
                pass

            finished_seqs = self.scheduler.postprocess(output.scheduled_seqs, token_ids)
            for seq in finished_seqs:
                # TODO: free PagedCache and collect RequestOutput
                pass

            return [self.outputs[seq.seq_id] for seq in finished_seqs]


def main():
    # Task 4.2.2: Initialize configs
    model_cfg = ModelConfig(
        # TODO: TinyLlama parameters
    )
    cache_cfg = CacheConfig(
        # TODO
    )
    scheduler_cfg = SchedulerConfig(
        # TODO: set max_seqs_in_flight and other parameters
    )

    engine = RealLLMEngine(
        model_name=MODEL_NAME,
        model_config=model_cfg,
        cache_config=cache_cfg,
        scheduler_config=scheduler_cfg,
        device=DEVICE,
    )

    tokenizer = engine.model_runner.tokenizer

    # Task 4.2.3: Submit multiple requests
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
