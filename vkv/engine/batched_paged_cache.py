"""
Phase 8, Part 1: BatchedPagedCache

Extends PagedCache to support multiple sequences in a single forward pass.
Each batch index is a distinct sequence with its own block_table and
_seq_length. K/V input/output shapes carry a real batch dimension.

Enables true continuous batching in RealLLMEngine.step().
"""

from typing import List, Optional, Tuple

import torch

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = object

from vkv.engine.block_manager import BlockManager


class BatchedPagedCache(Cache):
    """
    Paged KV cache that handles a batch of sequences.

    Shape conventions:
      - K/V input to update():  [batch, num_kv_heads, new_tokens, head_dim]
                                 batch == number of active sequences
                                 new_tokens usually == 1 (decode) or prompt_len (prefill)
      - K/V output from update(): [batch, num_kv_heads, max_seq_len, head_dim]
                                   padded so all sequences align

    Per-sequence state:
      block_tables[i]:   physical block IDs owned by sequence i
      _seq_lengths[i]:   how many tokens sequence i has cached so far
    """

    def __init__(
        self,
        block_manager: BlockManager,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
    ):
        """
        TODO:
        1. Store block_manager, num_layers, num_kv_heads, head_dim, block_size
        2. Initialize block_tables: List[List[int]] = []
             (each inner list is one sequence's block table)
        3. Initialize _seq_lengths: List[int] = []
        """
        raise NotImplementedError("TODO: Implement BatchedPagedCache.__init__")

    # ─────────────────────────────────────────────
    # Sequence lifecycle: add / remove batched entries
    # ─────────────────────────────────────────────

    def add_sequence(self, block_table: List[int], seq_length: int) -> int:
        """
        Register a new sequence in the batch (after prefill).

        Returns:
            batch_idx assigned to this sequence.

        TODO:
        1. Append block_table and seq_length to self.block_tables/_seq_lengths
        2. Return the batch index (== len(self.block_tables) - 1 after append)
        """
        raise NotImplementedError

    def remove_sequence(self, batch_idx: int):
        """Remove a finished sequence from the batch.

        TODO: pop block_tables[batch_idx] and _seq_lengths[batch_idx].
        Do NOT free blocks here (the caller frees them via block_manager).
        """
        raise NotImplementedError

    @property
    def batch_size(self) -> int:
        return len(self.block_tables)

    # ─────────────────────────────────────────────
    # HF Cache interface
    # ─────────────────────────────────────────────

    def update(
        self,
        key_states: torch.Tensor,     # [B, H, T_new, D]
        value_states: torch.Tensor,   # [B, H, T_new, D]
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Write new K/V for each sequence, then return padded full K/V.

        TODO:
        1. B, H, T_new, D = key_states.shape
        2. For each batch_idx in range(B):
             For each i in range(T_new):
               token_pos = self._seq_lengths[batch_idx] + i
               block_idx = token_pos // self.block_size
               slot_idx  = token_pos %  self.block_size
               if block_idx >= len(self.block_tables[batch_idx]):
                   new_id = self.block_manager.allocate(1)[0]
                   self.block_tables[batch_idx].append(new_id)
               K_slice = key_states[batch_idx, :, i, :]
               V_slice = value_states[batch_idx, :, i, :]
               self.block_manager.write_kv(
                   self.block_tables[batch_idx][block_idx], layer_idx, slot_idx,
                   K_slice, V_slice,
               )

        3. Update _seq_lengths ONLY on the last layer (avoid double count):
             if layer_idx + 1 == self.num_layers:
               for b in range(B):
                   self._seq_lengths[b] += T_new

        4. Gather + pad K/V for each seq → stacked [B, H, max_seq_len, D]:
             Use _pad_and_stack_kv(...) helper.

        5. Return (full_keys, full_values).
        """
        raise NotImplementedError("TODO: Implement BatchedPagedCache.update")

    def _pad_and_stack_kv(
        self, layer_idx: int, num_tokens_per_seq: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each sequence, gather its KV and pad to max_seq_len.

        TODO:
        1. max_seq_len = max(num_tokens_per_seq)
        2. For each batch_idx:
             k_seq, v_seq = block_manager.gather_kv(
                 block_tables[batch_idx], num_tokens_per_seq[batch_idx], layer_idx
             )
             # k_seq shape: [1, H, seq_len_i, D]
             # Pad on time dim to max_seq_len (torch.nn.functional.pad or manual zeros)
        3. torch.cat all padded tensors along dim=0 → [B, H, max_seq_len, D]
        4. Return (keys_padded, values_padded)
        """
        raise NotImplementedError

    # ─────────────────────────────────────────────
    # Helpers required by HF's mask logic
    # ─────────────────────────────────────────────

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the max seq_length across all sequences.

        TODO: return max(self._seq_lengths) if self._seq_lengths else 0
        """
        raise NotImplementedError

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple:
        """HF attention-mask helper.

        TODO:
        - if not self._seq_lengths (empty cache): return (query_length, 0)
        - else: return (max(self._seq_lengths), 0)
        """
        raise NotImplementedError

    def free(self):
        """Free every sequence's blocks.

        TODO:
        for bt in self.block_tables:
            self.block_manager.free(bt)
        self.block_tables.clear()
        self._seq_lengths.clear()
        """
        raise NotImplementedError
