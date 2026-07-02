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

from transformers import AutoModelForCausalLM, AutoTokenizer

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

        1. Import and load AutoModelForCausalLM and AutoTokenizer
        2. Extract model config → create ModelConfig
        3. Store block_manager reference
        4. Store block_size from block_manager
        """
        self.device = device
        self.dtype = dtype
        self.block_manager = block_manager
        self.block_size = block_manager.block_size
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_config = self._extract_model_config()


    def _extract_model_config(self) -> ModelConfig:
        """
        Extract ModelConfig from HuggingFace model config.
        """

        cfg = self.model.config
        return ModelConfig(
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
            dtype=self.dtype,
        )

    @torch.inference_mode()
    def prefill(self, input_ids: List[int]) -> PagedCache:
        """
        Run prefill: compute KV cache for the entire prompt.

        Args:
            input_ids: Prompt token IDs

        Returns:
            PagedCache containing the computed KV cache

        1. Create PagedCache instance
        2. Convert input_ids to tensor
        3. Run model forward: model(input_ids, past_key_values=paged_cache)
           The model internally calls paged_cache.update() for each layer,
           which writes KV data to BlockManager.
        4. Return paged_cache
        """
        paged_cache = PagedCache(
            self.block_manager, 
            self.model_config.num_layers, 
            self.model_config.num_kv_heads,
            self.model_config.head_dim,
            self.block_size
            )
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        output = self.model(input_tensor, past_key_values=paged_cache)
        logits = output.logits[:, -1, :]

        return paged_cache

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

        1. Convert token_id to tensor [1, 1]
        2. Run model forward with past_key_values=paged_cache
        3. Return (logits, paged_cache)
        """
        input_tensor = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        output = self.model.forward(input_tensor, past_key_values=paged_cache)
        logits = output.logits[:, 0, :]
        return (logits, paged_cache)

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

        1. Apply temperature: logits = logits / temperature
        2. Convert to probabilities: probs = softmax(logits)
        3. Sample: token_id = torch.multinomial(probs, 1)
        4. Return token_id as int
        """
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, 1).item()
        return token_id

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> str:
        """
        Convenience method: full generation from text prompt to text output.

        1. Tokenize prompt
        2. Prefill → get paged_cache
        3. Loop decode_step + sample for max_new_tokens
        4. Decode output tokens to text
        5. Free paged_cache
        6. Return text
        """
        input_ids = self.tokenizer.encode(prompt)
        paged_cache = self.prefill(input_ids)

        generated = []
        token_id = input_ids[-1]
        for _ in range(max_new_tokens):
            logits, paged_cache = self.decode_step(token_id, paged_cache)
            token_id = self.sample(logits, temperature)
            generated.append(token_id)

        paged_cache.free()

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.replace("<|assistant|>", "").replace("<|user|>", "").strip()

