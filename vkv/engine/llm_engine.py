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
from typing import Dict, List, Optional

from vkv.config import ModelConfig, CacheConfig
from vkv.sampling_params import SamplingParams
from vkv.engine.block_manager import BlockManager
from vkv.engine.sequence import Sequence, SequenceStatus
from vkv.engine.scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from vkv.engine.model_runner import MockModelRunner


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
        raise NotImplementedError("TODO: Implement LLMEngine.step")

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

        TODO: Implement this.
        1. If prompts provided, add_request for each
        2. Loop step() until is_finished()
        3. Collect and return all outputs, sorted by seq_id
        """
        raise NotImplementedError("TODO: Implement LLMEngine.generate")
