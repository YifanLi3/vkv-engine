"""
Phase 6, Part 1 & 3: RealModelRunner — runs actual HuggingFace models.

Replaces MockModelRunner with real model computation.
Uses PagedCache for BlockManager-backed KV cache.
"""

from typing import List, Optional, Tuple

import torch

from vkv.config import ModelConfig, CacheConfig
from vkv.engine.block_manager import BlockManager
from vkv.engine.paged_cache import PagedCache


class RealModelRunner:
    """
    Runs real HuggingFace model inference with vkv-engine's KV cache management.

    Matches nano-vLLM's ModelRunner role:
      - Load model weights
      - Run prefill and decode forward passes
      - Sample next token

    Usage:
        >>> runner = RealModelRunner("TinyLlama/TinyLlama-1.1B-Chat-v1.0", block_manager)
        >>> runner.prefill(seq)     # compute KV for prompt, write to BlockManager
        >>> token_id = runner.decode_step(seq)  # generate one token
    """

    def __init__(
        self,
        model_name: str,
        block_manager: BlockManager,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Load model and tokenizer.

        Args:
            model_name: HuggingFace model name or path
            block_manager: BlockManager for KV cache storage
            device: "cuda" or "cpu"
            dtype: Model dtype (float16 for GPU, float32 for CPU)

        TODO: Implement this.
        1. Import and load AutoModelForCausalLM and AutoTokenizer
        2. Extract model config → create ModelConfig
        3. Store block_manager reference
        4. Store block_size from block_manager
        """
        raise NotImplementedError("TODO: Implement RealModelRunner.__init__")

    def _extract_model_config(self) -> ModelConfig:
        """
        Extract ModelConfig from HuggingFace model config.

        TODO: Implement this.
        cfg = self.model.config
        return ModelConfig(
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
            dtype=self.dtype,
        )
        """
        raise NotImplementedError("TODO: Implement _extract_model_config")

    @torch.inference_mode()
    def prefill(self, input_ids: List[int]) -> PagedCache:
        """
        Run prefill: compute KV cache for the entire prompt.

        Args:
            input_ids: Prompt token IDs

        Returns:
            PagedCache containing the computed KV cache

        TODO: Implement this.
        1. Create PagedCache instance
        2. Convert input_ids to tensor
        3. Run model forward: model(input_ids, past_key_values=paged_cache)
           The model internally calls paged_cache.update() for each layer,
           which writes KV data to BlockManager.
        4. Return paged_cache
        """
        raise NotImplementedError("TODO: Implement RealModelRunner.prefill")

    @torch.inference_mode()
    def decode_step(
        self,
        token_id: int,
        paged_cache: PagedCache,
    ) -> Tuple[torch.Tensor, PagedCache]:
        """
        Run one decode step: generate next token's KV and logits.

        Args:
            token_id: Last generated token ID
            paged_cache: Existing KV cache from prefill or previous decode

        Returns:
            (logits, updated_paged_cache)
            logits shape: [1, vocab_size]

        TODO: Implement this.
        1. Convert token_id to tensor [1, 1]
        2. Run model forward with past_key_values=paged_cache
        3. Return (logits, paged_cache)
        """
        raise NotImplementedError("TODO: Implement RealModelRunner.decode_step")

    def sample(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> int:
        """
        Sample next token from logits.

        Args:
            logits: [1, vocab_size] or [vocab_size]
            temperature: Sampling temperature

        Returns:
            Sampled token ID

        TODO: Implement this.
        1. Apply temperature: logits = logits / temperature
        2. Convert to probabilities: probs = softmax(logits)
        3. Sample: token_id = torch.multinomial(probs, 1)
        4. Return token_id as int
        """
        raise NotImplementedError("TODO: Implement RealModelRunner.sample")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> str:
        """
        Convenience method: full generation from text prompt to text output.

        TODO: Implement this.
        1. Tokenize prompt
        2. Prefill → get paged_cache
        3. Loop decode_step + sample for max_new_tokens
        4. Decode output tokens to text
        5. Free paged_cache
        6. Return text
        """
        raise NotImplementedError("TODO: Implement RealModelRunner.generate")
