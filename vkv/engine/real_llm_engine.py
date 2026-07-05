"""
RealLLMEngine — LLMEngine with RealModelRunner replacing MockModelRunner.

Extends LLMEngine.step() to drive real HuggingFace model inference with
PagedCache-backed KV storage.

Key differences from LLMEngine:
  - Maintains paged_caches dict: seq_id -> PagedCache
  - prefill: runner.prefill(prompt_ids) -> PagedCache (KV written internally)
  - decode:  runner.decode_step(last_token, paged_cache) -> (logits, cache)
  - on finish: paged_cache.free()
"""

from typing import Dict, List

from vkv.config import ModelConfig, CacheConfig
from vkv.engine.block_manager import BlockManager
from vkv.engine.llm_engine import LLMEngine, RequestOutput
from vkv.engine.paged_cache import PagedCache
from vkv.engine.real_model_runner import RealModelRunner
from vkv.engine.scheduler import SchedulerConfig


class RealLLMEngine(LLMEngine):

    def __init__(
        self,
        model_name: str,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig = None,
        device: str = "cuda",
    ):
        super().__init__(model_config, cache_config, scheduler_config, device)

        self.model_runner = RealModelRunner(
            model_name=model_name,
            block_manager=BlockManager(model_config, cache_config, device),
            device=device,
        )

        # One PagedCache per active sequence
        self.paged_caches: Dict[int, PagedCache] = {}

    def step(self) -> List[RequestOutput]:
        output = self.scheduler.schedule()

        if output.is_prefill:
            for seq in output.scheduled_seqs:
                paged_cache = self.model_runner.prefill(
                    seq.token_ids[:seq.num_prompt_tokens]
                )
                self.paged_caches[seq.seq_id] = paged_cache
            return []

        token_ids = []
        for seq in output.scheduled_seqs:
            last_token_id = seq.token_ids[-1]
            paged_cache = self.paged_caches[seq.seq_id]
            logits, paged_cache = self.model_runner.decode_step(last_token_id, paged_cache)
            temperature = seq.sampling_params.temperature if seq.sampling_params else 1.0
            token_id = self.model_runner.sample(logits, temperature)
            token_ids.append(token_id)

        finished_seqs = self.scheduler.postprocess(output.scheduled_seqs, token_ids)
        for seq in finished_seqs:
            self.paged_caches[seq.seq_id].free()
            self.outputs[seq.seq_id] = RequestOutput(
                seq_id=seq.seq_id,
                prompt_token_ids=seq.token_ids[:seq.num_prompt_tokens],
                output_token_ids=seq.token_ids[seq.num_prompt_tokens:],
            )

        return [self.outputs[seq.seq_id] for seq in finished_seqs]
