"""
Phase 2, Part 5: LLMEngine — top-level coordinator.

Naming aligned with nano-vLLM/vLLM:
  nano-vLLM: LLMEngine (llm_engine.py) — wires scheduler + model_runner
  vLLM:      LLMEngine (llm_engine.py) — same role
  vkv:       LLMEngine (this file)

nano-vLLM's LLMEngine core:
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

    def generate(self, prompts, sampling_params):
        for prompt in prompts:
            self.add_request(prompt, sp)
        while not self.is_finished():
            self.step()
"""

from dataclasses import dataclass, field
import token
from typing import Dict, List, Optional

from vkv.config import ModelConfig, CacheConfig
from vkv.sampling_params import SamplingParams
from vkv.engine.block_manager import BlockManager
from vkv.engine.sequence import Sequence, SequenceStatus
from vkv.engine.scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from vkv.engine.model_runner import MockModelRunner

from vkv.engine.monitor import Monitor


@dataclass
class RequestOutput:
    """Output for a completed request. Provided for you."""
    seq_id: int
    prompt_token_ids: List[int]
    output_token_ids: List[int]


class LLMEngine:
    """
    Top-level engine that coordinates Scheduler + ModelRunner + BlockManager.

    Matches nano-vLLM's LLMEngine interface:
        add_request(prompt, sampling_params)
        step() → one round of schedule + execute + postprocess
        generate(prompts) → run to completion

    Usage:
        >>> engine = LLMEngine(model_config, cache_config, scheduler_config)
        >>> engine.add_request([1, 2, 3, 4])
        >>> engine.add_request([5, 6, 7])
        >>> outputs = engine.generate()
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig = None,
        device: str = "cpu",
    ):
        """
        Initialize the engine.

        Creates BlockManager, Scheduler, and MockModelRunner.

        nano-vLLM's LLMEngine.__init__ does:
            self.model_runner = ModelRunner(config, ...)
            self.scheduler = Scheduler(config)

        1. Store configs
        2. Create BlockManager(model_config, cache_config, device)
        3. Create Scheduler(block_manager, scheduler_config)
        4. Create MockModelRunner(model_config, device)
        5. Initialize outputs dict: Dict[int, RequestOutput]
        """
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config

        self.block_manager = BlockManager(model_config, cache_config, device)
        self.scheduler = Scheduler(self.block_manager, scheduler_config)
        self.model_runner = MockModelRunner(model_config, device)
        self.outputs: Dict[int, RequestOutput] = {}
        self.monitor = Monitor()

    def add_request(
        self,
        token_ids: List[int],
        sampling_params: SamplingParams = None,
    ) -> int:
        """
        Add a new request to the engine.

        Matches nano-vLLM's LLMEngine.add_request():
            seq = Sequence(prompt, sampling_params)
            self.scheduler.add(seq)

        Args:
            token_ids: Prompt token IDs
            sampling_params: Generation parameters

        Returns:
            seq_id: Unique ID for this request

        1. Create Sequence(token_ids, block_manager, sampling_params)
        2. scheduler.add(seq)
        3. Return seq.seq_id
        """
        seq = Sequence(token_ids, self.block_manager, sampling_params)
        self.scheduler.add(seq)
        self.monitor.on_request_arrival(seq.seq_id, len(token_ids))
        return seq.seq_id

    def step(self) -> List[RequestOutput]:
        """
        Execute one step: schedule → run model → postprocess.

        Matches nano-vLLM's LLMEngine.step():
            seqs, is_prefill = self.scheduler.schedule()
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids)

        Since we use MockModelRunner, "running the model" means generating
        random token IDs. The important part is the scheduler logic.

        Returns:
            List of RequestOutputs for sequences that finished this step.

        1. Call scheduler.schedule() → get SchedulerOutput
        2. If is_prefill:
           - For each seq, write KV data to BlockManager
             (using MockModelRunner.prefill to generate random KV)
           - No tokens generated in prefill step (return [])
        3. If decode:
           - For each seq, generate a random token (MockModelRunner.sample)
           - Call scheduler.postprocess(seqs, token_ids)
           - Collect finished sequences into RequestOutput list
        4. Return finished outputs
        """
        output = self.scheduler.schedule()
        if output.is_prefill:
            for seq in output.scheduled_seqs:
                kv = self.model_runner.prefill(len(seq))
                for layer in range(self.model_config.num_layers):
                    for token_pos in range(seq.num_prompt_tokens):
                        block_id = seq.block_table[token_pos // self.block_manager.block_size]
                        slot_idx = token_pos % self.block_manager.block_size
                        self.block_manager.write_kv(
                            block_id, layer, slot_idx,
                            kv[layer, 0, :, token_pos, :],
                            kv[layer, 1, :, token_pos, :],
                        )
            for seq in output.scheduled_seqs:
                self.monitor.on_first_token(seq.seq_id)
            self.monitor.on_step(self.block_manager, self.scheduler)
            return []
        else:
            token_ids = []
            for seq in output.scheduled_seqs:
                token_id = self.model_runner.sample()
                token_ids.append(token_id)

            finished_seqs = self.scheduler.postprocess(output.scheduled_seqs, token_ids)
            for seq in finished_seqs:
                num_output = len(seq.token_ids) - seq.num_prompt_tokens
                self.monitor.on_request_finish(seq.seq_id, num_output)
                self.outputs[seq.seq_id] = RequestOutput(
                    seq_id=seq.seq_id,
                    prompt_token_ids=seq.token_ids[:seq.num_prompt_tokens],
                    output_token_ids=seq.token_ids[seq.num_prompt_tokens:],
                )
            self.monitor.on_step(self.block_manager, self.scheduler)
            return [self.outputs[seq.seq_id] for seq in finished_seqs]


    def is_finished(self) -> bool:
        """Check if all requests are done. Delegates to scheduler."""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: Optional[List[List[int]]] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        """
        Run to completion: add requests + step until all done.

        Matches nano-vLLM's LLMEngine.generate():
            for prompt in prompts:
                self.add_request(prompt, sp)
            while not self.is_finished():
                self.step()

        Args:
            prompts: List of token ID lists (if None, assumes requests already added)
            sampling_params: Generation params (applied to all requests)

        Returns:
            List of RequestOutput for all completed requests.

        1. If prompts provided, add_request for each
        2. Loop step() until is_finished()
        3. Collect and return all outputs, sorted by seq_id
        """
        if prompts:
            for prompt in prompts:
                self.add_request(prompt, sampling_params)

        while not self.is_finished():
            self.step()

        return sorted(self.outputs.values(), key=lambda x: x.seq_id)
