"""
Phase 4: Prefix Caching — Radix Tree + Copy-on-Write

Reduces redundant KV computation by caching and reusing KV blocks for
shared prompt prefixes across requests.

Inspired by:
  SGLang RadixAttention (https://arxiv.org/abs/2312.07104)
  vLLM automatic prefix caching

No direct nano-vLLM equivalent — nano-vLLM uses a simple hash-based block
cache; our implementation uses a full Radix Tree for exact prefix matching.
"""

import time
from typing import Dict, List, Optional, Tuple

from vkv.engine.block_manager import BlockManager


# =============================================================================
# Part 1: RadixNode — the tree building block
# =============================================================================

class RadixNode:
    """
    A node in the Radix Tree representing a cached token sequence segment.

    Each node holds a contiguous segment of tokens and the physical KV cache
    blocks that store those tokens' K/V data. Children are keyed by the first
    token of their segment, enabling O(prefix_len) lookup.

    Example tree (block_size=2, sequences [1,2,3,4] and [1,2,5,6]):

        root  (tokens=[], blocks=[])
          └─ key=1 → node(tokens=[1,2], blocks=[3])
                        ├─ key=3 → node(tokens=[3,4], blocks=[7])
                        └─ key=5 → node(tokens=[5,6], blocks=[9])

    Attributes:
        token_ids:   The token segment this node covers (relative to parent)
        block_ids:   Physical block IDs holding KV for token_ids
        children:    Dict[first_token → child RadixNode]
        parent:      Reference to parent node (None for root)
        last_access: Monotonic timestamp — updated on every cache hit (for LRU)
        ref_count:   How many active sequences are currently using this node's
                     blocks. Nodes with ref_count > 0 must NOT be evicted.
    """

    def __init__(
        self,
        token_ids: Optional[List[int]] = None,
        block_ids: Optional[List[int]] = None,
        parent: Optional['RadixNode'] = None,
    ):
        self.token_ids: List[int] = token_ids if token_ids is not None else []
        self.block_ids: List[int] = block_ids if block_ids is not None else []
        self.children: Dict[int, 'RadixNode'] = {}
        self.parent: Optional['RadixNode'] = parent
        self.last_access: float = time.monotonic()
        self.ref_count: int = 0

    @property
    def is_leaf(self) -> bool:
        """True if this node has no children."""
        return len(self.children) == 0

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    def __repr__(self) -> str:
        return (
            f"RadixNode(tokens={self.token_ids}, blocks={self.block_ids}, "
            f"children={list(self.children.keys())}, ref={self.ref_count})"
        )


# =============================================================================
# Part 2–6: PrefixCache — Radix Tree with LRU eviction and COW
# =============================================================================

class PrefixCache:
    """
    Radix Tree-based KV prefix cache.

    Caches KV blocks for completed token blocks and reuses them when new
    requests share the same prefix — avoiding redundant prefill computation.

    Typical workflow:
        # 1. Before prefill — check for a cached prefix
        matched_blocks, num_cached = cache.match_prefix(prompt_token_ids)
        # matched_blocks: blocks to reuse; skip prefill for first num_cached tokens

        # 2. After prefill completes — store the result
        cache.insert(prompt_token_ids, seq.block_table)

        # 3. When GPU memory is low — free unused cached blocks
        freed = cache.evict(num_blocks_needed)

        # 4. Before writing to a shared block — copy it first (COW)
        if block_manager.get_ref_count(block_id) > 1:
            block_id = cache.copy_on_write(block_id)
            seq.block_table[i] = block_id

    Alignment rule:
        Only COMPLETE blocks are cached (len(token_ids) must be a multiple of
        block_size at insertion). Partial last blocks are excluded because they
        may still grow during the decode phase.
    """

    def __init__(self, block_manager: BlockManager, block_size: int):
        """
        Args:
            block_manager: BlockManager for block allocation / freeing / copying
            block_size:    Tokens per block (from CacheConfig)
        """
        self.block_manager = block_manager
        self.block_size = block_size
        self.root = RadixNode()

        self._hits = 0
        self._misses = 0

    # -------------------------------------------------------------------------
    # Part 2: match_prefix
    # -------------------------------------------------------------------------

    def match_prefix(
        self,
        token_ids: List[int],
    ) -> Tuple[List[int], int]:
        """
        Find the longest cached prefix that matches the start of token_ids.

        Walk the radix tree greedily from the root. At each node check whether
        the next tokens in token_ids match the node's token_ids. Continue
        until no further child matches or the sequence is exhausted.

        Only tokens at completed block boundaries count — num_matched_tokens
        is always a multiple of block_size.

        Args:
            token_ids: The full token sequence for the incoming request.

        Returns:
            (matched_block_ids, num_matched_tokens)
            matched_block_ids:   Ordered physical block IDs to reuse (may be [])
            num_matched_tokens:  Tokens covered by matched_block_ids (may be 0)

        Example (block_size=2):
            Cached: [1,2,3,4] → blocks [3, 7]
            Query:  [1,2,3,4,5,6]
            → returns ([3, 7], 4)

        Algorithm:
            matched_blocks = []
            cursor = 0  # index into token_ids
            node = self.root

            loop:
                if cursor >= len(token_ids): break
                key = token_ids[cursor]
                if key not in node.children: break
                child = node.children[key]
                common = _match_len(child.token_ids, token_ids[cursor:])
                if common == child.num_tokens:          # full node matched
                    matched_blocks += child.block_ids
                    cursor += child.num_tokens
                    child.last_access = time.monotonic()
                    node = child
                else:                                   # partial match — stop
                    break

            update _hits / _misses
            return matched_blocks, cursor
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Part 3: insert
    # -------------------------------------------------------------------------

    def insert(
        self,
        token_ids: List[int],
        block_ids: List[int],
    ) -> None:
        """
        Insert a computed token sequence and its KV blocks into the cache.

        Only complete blocks are stored. If the last block is partial (i.e.
        len(token_ids) % block_size != 0), it is silently ignored.

        Args:
            token_ids: Token sequence covering exactly the blocks in block_ids.
                       len(token_ids) should equal len(block_ids) * block_size.
            block_ids: Ordered physical block IDs (one per block_size tokens).

        Alignment rule applied internally:
            num_complete = len(token_ids) // block_size
            tokens_to_cache = token_ids[:num_complete * block_size]
            blocks_to_cache = block_ids[:num_complete]

        Algorithm (radix tree insertion with node splitting):
            node = root; cursor = 0; block_cursor = 0

            while cursor < len(tokens_to_cache):
                key = tokens_to_cache[cursor]

                if key NOT in node.children:
                    # No existing child — create a new leaf and stop
                    new_node = RadixNode(
                        token_ids=tokens_to_cache[cursor:],
                        block_ids=blocks_to_cache[block_cursor:],
                        parent=node,
                    )
                    node.children[key] = new_node
                    # Increment ref on each block (shared ownership)
                    for bid in new_node.block_ids:
                        block_manager.inc_ref(bid)
                    break

                child = node.children[key]
                common_raw = _match_len(child.token_ids, tokens_to_cache[cursor:])

                # IMPORTANT: a block is the smallest cacheable unit. Even if
                # common_raw (token-level match) isn't a multiple of
                # block_size, we can only actually SHARE whole blocks — a
                # block is physically identical only if every token inside
                # it matches. So round the match length DOWN to the nearest
                # block boundary before doing anything else:
                #
                #   common_blocks = common_raw // block_size
                #   common = common_blocks * block_size   # use this, not common_raw
                #
                # Example (block_size=2): child=[1,2,3,4], new=[1,2,3,5]
                #   common_raw = 3  (tokens 1,2,3 match)
                #   common_blocks = 3 // 2 = 1
                #   common = 1 * 2 = 2   ← only block [1,2] is truly shared;
                #                          block [3,4] vs [3,5] differ, so
                #                          token 3 alone does NOT count.
                common_blocks = common_raw // block_size
                common = common_blocks * block_size

                if common == child.num_tokens:
                    # Full match — descend into child and continue
                    cursor += child.num_tokens
                    block_cursor += child.num_blocks
                    node = child

                elif common == 0:
                    # No block-aligned overlap at all (e.g. common_raw < block_size).
                    # Nothing can be shared here — stop without modifying the tree.
                    break

                else:
                    # Partial match (0 < common < child.num_tokens) — SPLIT
                    # the existing child at the block-aligned boundary `common`.
                    #
                    # Before split (common=2, block_size=2):
                    #   node → child([1,2,3,4], blocks=[A,B])
                    #
                    # After split:
                    #   node → mid([1,2], blocks=[A])
                    #               ├─ suffix([3,4], blocks=[B])   ← old child tail
                    #               └─ new([5,6], blocks=[C])      ← new tokens
                    #
                    # Steps (all slicing uses the block-aligned `common`,
                    # never common_raw):
                    # 1. Create mid_node with child.token_ids[:common] and child.block_ids[:common_blocks]
                    # 2. Create suffix_node = old child trimmed to
                    #    token_ids[common:] and block_ids[common_blocks:]
                    # 3. mid_node.children[suffix_first_token] = suffix_node
                    # 4. node.children[key] = mid_node
                    # 5. Create new leaf under mid_node for the remaining
                    #    tokens_to_cache[cursor+common:] (this may still need
                    #    its own alignment check if it's shorter than block_size)
                    # 6. Increment ref on new leaf's blocks
                    # (mid and suffix already share the same physical blocks — no extra inc_ref needed)
                    break  # TODO: implement splitting

        Hint: Always slice both token_ids and block_ids using the SAME
        block-aligned `common` / `common_blocks` pair. Never slice token_ids
        with the raw (unaligned) match length — that desyncs a node's
        token_ids from its block_ids, since len(token_ids) must always equal
        len(block_ids) * block_size.

        After a split, the mid_node inherits the physical blocks for the
        common prefix from the original child (no extra inc_ref needed for
        those). Only the newly inserted leaf's blocks need inc_ref.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Part 4: evict
    # -------------------------------------------------------------------------

    def evict(self, num_blocks: int) -> int:
        """
        Free at least num_blocks cached blocks using LRU eviction.

        Only leaf nodes with ref_count == 0 are eligible for eviction.
        Evict the leaf with the oldest last_access first.

        After evicting a leaf, its parent may become a new leaf — it can be
        considered for eviction in the same call if still ref_count == 0.

        Args:
            num_blocks: Target number of blocks to free (evict until met).

        Returns:
            Number of blocks actually freed (may be < num_blocks if cache is
            too small or all remaining blocks are in use).

        Algorithm:
            freed = 0
            while freed < num_blocks:
                leaves = all leaf nodes with ref_count == 0 (use _collect_leaves)
                if no leaves: break
                victim = leaf with smallest last_access (oldest)
                block_manager.free(victim.block_ids)
                freed += victim.num_blocks
                del victim.parent.children[victim.token_ids[0]]
                # parent might now be a leaf — loop will catch it next iteration
            return freed
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Part 5: Copy-on-Write
    # -------------------------------------------------------------------------

    def copy_on_write(self, block_id: int) -> int:
        """
        COW: allocate a fresh block, copy KV data from the shared block.

        Call this before writing new tokens into a block that is shared with
        the prefix cache (ref_count > 1). After COW, the sequence owns a
        private copy and the cached block is untouched.

        Args:
            block_id: The shared block to copy from.

        Returns:
            new_block_id: A new block with identical KV data, ref_count == 1.

        Steps:
            1. new_block_id = block_manager.allocate(1)[0]
            2. block_manager.copy_block(block_id, new_block_id)
            3. block_manager.free([block_id])   # decrement ref on the original
            4. return new_block_id
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Part 6: Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Return a snapshot of cache statistics.

        Returns a dict with keys:
            hit_rate:             float   — hits / (hits + misses); 0.0 if no queries yet
            total_hits:           int
            total_misses:         int
            total_cached_blocks:  int     — blocks currently stored in the tree
            num_nodes:            int     — tree nodes (excluding root)

        Hint: use _count_nodes_and_blocks(self.root) for tree totals.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Internal helpers — provided, do not modify
    # -------------------------------------------------------------------------

    def _match_len(self, a: List[int], b: List[int]) -> int:
        """Return the length of the common prefix shared by lists a and b."""
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    def _collect_leaves(self, node: RadixNode, result: List[RadixNode]) -> None:
        """Recursively collect all leaf nodes in the subtree rooted at node."""
        if node.is_leaf and node is not self.root:
            result.append(node)
        for child in node.children.values():
            self._collect_leaves(child, result)

    def _count_nodes_and_blocks(self, node: RadixNode) -> Tuple[int, int]:
        """Recursively count (nodes, blocks) in the subtree rooted at node."""
        num_nodes = 0 if node is self.root else 1
        num_blocks = len(node.block_ids)
        for child in node.children.values():
            n, b = self._count_nodes_and_blocks(child)
            num_nodes += n
            num_blocks += b
        return num_nodes, num_blocks
