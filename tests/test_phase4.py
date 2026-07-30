"""
Phase 4 Test Suite — Prefix Caching (Radix Tree + COW)

Run all tests:       uv run pytest tests/test_phase4.py -v
Run one Part:        uv run pytest tests/test_phase4.py -k "part2" -v
"""

import pytest

from vkv.config import CacheConfig, TINY_MODEL
from vkv.engine.block_manager import BlockManager


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def block_manager():
    """Small BlockManager: 4 layers, 4 heads, head_dim=64, 100 GPU blocks."""
    return BlockManager(
        TINY_MODEL,
        CacheConfig(block_size=4, num_gpu_blocks=100, num_cpu_blocks=10),
        device="cpu",
    )


@pytest.fixture
def cache(block_manager):
    from vkv.engine.prefix_cache import PrefixCache
    return PrefixCache(block_manager, block_size=4)


# =============================================================================
# Part 1: RadixNode — data structure
# =============================================================================

class TestPart1:
    """Verify RadixNode properties without touching PrefixCache logic."""

    def test_default_node_is_empty(self):
        from vkv.engine.prefix_cache import RadixNode
        node = RadixNode()
        assert node.token_ids == []
        assert node.block_ids == []
        assert node.children == {}
        assert node.parent is None
        assert node.ref_count == 0

    def test_is_leaf_no_children(self):
        from vkv.engine.prefix_cache import RadixNode
        node = RadixNode(token_ids=[1, 2], block_ids=[10])
        assert node.is_leaf is True

    def test_is_leaf_with_children(self):
        from vkv.engine.prefix_cache import RadixNode
        parent = RadixNode(token_ids=[1, 2], block_ids=[10])
        child = RadixNode(token_ids=[3, 4], block_ids=[11], parent=parent)
        parent.children[3] = child
        assert parent.is_leaf is False
        assert child.is_leaf is True

    def test_num_tokens_and_blocks(self):
        from vkv.engine.prefix_cache import RadixNode
        node = RadixNode(token_ids=[1, 2, 3, 4], block_ids=[10, 11])
        assert node.num_tokens == 4
        assert node.num_blocks == 2

    def test_root_is_empty(self, cache):
        assert cache.root.token_ids == []
        assert cache.root.block_ids == []
        assert cache.root.is_leaf is True

    def test_match_len_helper(self, cache):
        assert cache._match_len([1, 2, 3], [1, 2, 4]) == 2
        assert cache._match_len([1, 2], [1, 2, 3]) == 2
        assert cache._match_len([], [1, 2]) == 0
        assert cache._match_len([1, 2], [1, 2]) == 2


# =============================================================================
# Part 2: match_prefix
# =============================================================================

class TestPart2:
    """Tests for PrefixCache.match_prefix()."""

    def test_empty_cache_returns_nothing(self, cache):
        blocks, count = cache.match_prefix([1, 2, 3, 4])
        assert blocks == []
        assert count == 0

    def test_full_match_one_node(self, cache, block_manager):
        """Cache [1,2,3,4] → block [5]. Query [1,2,3,4] should return ([5], 4)."""
        from vkv.engine.prefix_cache import RadixNode
        # Manually build tree (bypassing insert for isolated testing)
        bid = block_manager.allocate(1)[0]
        node = RadixNode(token_ids=[1, 2, 3, 4], block_ids=[bid], parent=cache.root)
        cache.root.children[1] = node

        blocks, count = cache.match_prefix([1, 2, 3, 4])
        assert blocks == [bid]
        assert count == 4

    def test_partial_query_matches_full_node(self, cache, block_manager):
        """Cache [1,2,3,4]. Query [1,2,3,4,5,6] — match first 4 tokens."""
        from vkv.engine.prefix_cache import RadixNode
        bid = block_manager.allocate(1)[0]
        node = RadixNode(token_ids=[1, 2, 3, 4], block_ids=[bid], parent=cache.root)
        cache.root.children[1] = node

        blocks, count = cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
        assert blocks == [bid]
        assert count == 4

    def test_no_match_wrong_prefix(self, cache, block_manager):
        from vkv.engine.prefix_cache import RadixNode
        bid = block_manager.allocate(1)[0]
        node = RadixNode(token_ids=[1, 2, 3, 4], block_ids=[bid], parent=cache.root)
        cache.root.children[1] = node

        blocks, count = cache.match_prefix([9, 8, 7, 6])
        assert blocks == []
        assert count == 0

    def test_two_level_match(self, cache, block_manager):
        """Two-level tree: root → [1,2,3,4] → [5,6,7,8]. Full match of 8 tokens."""
        from vkv.engine.prefix_cache import RadixNode
        b1, b2 = block_manager.allocate(2)
        n1 = RadixNode(token_ids=[1, 2, 3, 4], block_ids=[b1], parent=cache.root)
        n2 = RadixNode(token_ids=[5, 6, 7, 8], block_ids=[b2], parent=n1)
        cache.root.children[1] = n1
        n1.children[5] = n2

        blocks, count = cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert blocks == [b1, b2]
        assert count == 8

    def test_hit_miss_tracking(self, cache, block_manager):
        from vkv.engine.prefix_cache import RadixNode
        bid = block_manager.allocate(1)[0]
        node = RadixNode(token_ids=[1, 2, 3, 4], block_ids=[bid], parent=cache.root)
        cache.root.children[1] = node

        cache.match_prefix([1, 2, 3, 4])   # hit
        cache.match_prefix([9, 8, 7, 6])   # miss

        stats = cache.get_stats()
        assert stats["total_hits"] == 1
        assert stats["total_misses"] == 1

    def test_returned_count_is_multiple_of_block_size(self, cache, block_manager):
        from vkv.engine.prefix_cache import RadixNode
        b1, b2 = block_manager.allocate(2)
        n1 = RadixNode(token_ids=[1, 2, 3, 4], block_ids=[b1], parent=cache.root)
        n2 = RadixNode(token_ids=[5, 6, 7, 8], block_ids=[b2], parent=n1)
        cache.root.children[1] = n1
        n1.children[5] = n2

        _, count = cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert count % cache.block_size == 0


# =============================================================================
# Part 3: insert
# =============================================================================

class TestPart3:
    """Tests for PrefixCache.insert()."""

    def test_insert_single_block(self, cache, block_manager):
        bids = block_manager.allocate(1)
        cache.insert([1, 2, 3, 4], bids)

        blocks, count = cache.match_prefix([1, 2, 3, 4])
        assert count == 4
        assert blocks == bids

    def test_insert_two_blocks(self, cache, block_manager):
        bids = block_manager.allocate(2)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8], bids)

        blocks, count = cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
        assert count == 8
        assert blocks == bids

    def test_insert_ignores_partial_last_block(self, cache, block_manager):
        """9 tokens with block_size=4 → only 8 tokens (2 blocks) cached."""
        bids = block_manager.allocate(3)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8, 9], bids)

        _, count = cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert count == 8  # partial block not cached

    def test_shared_prefix_two_sequences(self, cache, block_manager):
        """Two sequences with common prefix [1,2,3,4] — should share that block."""
        bids_a = block_manager.allocate(2)
        bids_b = block_manager.allocate(2)
        # bids_b[0] might differ from bids_a[0] — the tree should deduplicate the prefix node
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8], bids_a)
        cache.insert([1, 2, 3, 4, 9, 10, 11, 12], bids_b)

        # Both lookups should find the common prefix
        blocks_a, count_a = cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
        blocks_b, count_b = cache.match_prefix([1, 2, 3, 4, 9, 10, 11, 12])

        assert count_a == 8
        assert count_b == 8
        # The first block of each result covers [1,2,3,4] — they should be the same
        assert blocks_a[0] == blocks_b[0]

    def test_insert_increases_node_count(self, cache, block_manager):
        bids = block_manager.allocate(1)
        cache.insert([1, 2, 3, 4], bids)
        stats = cache.get_stats()
        assert stats["num_nodes"] >= 1

    def test_insert_no_effect_if_empty_or_partial_only(self, cache, block_manager):
        """Inserting fewer than block_size tokens → nothing cached."""
        bids = block_manager.allocate(1)
        cache.insert([1, 2], bids)  # block_size=4, only 2 tokens → skip
        _, count = cache.match_prefix([1, 2, 3, 4])
        assert count == 0

    def test_insert_same_sequence_twice_is_idempotent(self, cache, block_manager):
        bids = block_manager.allocate(1)
        cache.insert([1, 2, 3, 4], bids)
        cache.insert([1, 2, 3, 4], bids)  # second insert should not crash

        blocks, count = cache.match_prefix([1, 2, 3, 4])
        assert count == 4

    def test_total_cached_blocks_after_insert(self, cache, block_manager):
        bids = block_manager.allocate(2)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8], bids)
        stats = cache.get_stats()
        assert stats["total_cached_blocks"] == 2


# =============================================================================
# Part 4: evict
# =============================================================================

class TestPart4:
    """Tests for PrefixCache.evict()."""

    def test_evict_single_leaf(self, cache, block_manager):
        bids = block_manager.allocate(2)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8], bids)

        before = block_manager.gpu_allocator.num_used
        freed = cache.evict(1)
        after = block_manager.gpu_allocator.num_used

        assert freed >= 1
        assert after < before

    def test_evict_returns_freed_count(self, cache, block_manager):
        bids = block_manager.allocate(2)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8], bids)
        freed = cache.evict(2)
        assert freed == 2

    def test_evict_on_empty_cache(self, cache):
        freed = cache.evict(5)
        assert freed == 0

    def test_evict_does_not_evict_in_use_nodes(self, cache, block_manager):
        from vkv.engine.prefix_cache import RadixNode
        bid = block_manager.allocate(1)[0]
        node = RadixNode(token_ids=[1, 2, 3, 4], block_ids=[bid], parent=cache.root)
        node.ref_count = 1  # marked as in use
        cache.root.children[1] = node

        freed = cache.evict(1)
        assert freed == 0
        assert block_manager.get_ref_count(bid) > 0  # block still allocated

    def test_evict_lru_order(self, cache, block_manager):
        """Older node should be evicted first."""
        import time
        bids_a = block_manager.allocate(1)
        bids_b = block_manager.allocate(1)

        cache.insert([1, 2, 3, 4], bids_a)
        time.sleep(0.01)  # ensure different timestamps
        cache.insert([5, 6, 7, 8], bids_b)

        # Evict 1 block — should be the older one (bids_a)
        freed = cache.evict(1)
        assert freed == 1

        # The newer sequence should still be findable
        blocks, count = cache.match_prefix([5, 6, 7, 8])
        assert count == 4

    def test_evict_cleans_up_node_from_parent(self, cache, block_manager):
        bids = block_manager.allocate(1)
        cache.insert([1, 2, 3, 4], bids)

        cache.evict(1)

        # After eviction, match should find nothing
        blocks, count = cache.match_prefix([1, 2, 3, 4])
        assert count == 0
        assert cache.root.is_leaf

    def test_evict_more_than_available(self, cache, block_manager):
        """Evicting more blocks than cached should not crash."""
        bids = block_manager.allocate(1)
        cache.insert([1, 2, 3, 4], bids)
        freed = cache.evict(100)
        assert freed == 1  # only 1 block was cached


# =============================================================================
# Part 5: copy_on_write
# =============================================================================

class TestPart5:
    """Tests for PrefixCache.copy_on_write()."""

    def test_cow_returns_new_block(self, cache, block_manager):
        bid = block_manager.allocate(1)[0]
        block_manager.inc_ref(bid)  # simulate cache also holding a ref (ref=2)

        new_bid = cache.copy_on_write(bid)
        assert new_bid != bid

    def test_cow_original_ref_decremented(self, cache, block_manager):
        bid = block_manager.allocate(1)[0]
        block_manager.inc_ref(bid)  # ref = 2

        cache.copy_on_write(bid)
        # Original should now have ref = 1 (cache kept it, sequence released it)
        assert block_manager.get_ref_count(bid) == 1

    def test_cow_new_block_has_ref_one(self, cache, block_manager):
        bid = block_manager.allocate(1)[0]
        block_manager.inc_ref(bid)  # ref = 2

        new_bid = cache.copy_on_write(bid)
        assert block_manager.get_ref_count(new_bid) == 1

    def test_cow_new_block_is_independent(self, cache, block_manager):
        """After COW, freeing new block should not affect the original."""
        bid = block_manager.allocate(1)[0]
        block_manager.inc_ref(bid)  # ref = 2

        new_bid = cache.copy_on_write(bid)
        block_manager.free([new_bid])  # free the copy

        # Original should still be alive (ref == 1 from cache)
        assert block_manager.get_ref_count(bid) == 1


# =============================================================================
# Part 6: get_stats
# =============================================================================

class TestPart6:
    """Tests for PrefixCache.get_stats()."""

    def test_stats_empty_cache(self, cache):
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0
        assert stats["total_hits"] == 0
        assert stats["total_misses"] == 0
        assert stats["total_cached_blocks"] == 0
        assert stats["num_nodes"] == 0

    def test_stats_after_insert(self, cache, block_manager):
        bids = block_manager.allocate(2)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8], bids)
        stats = cache.get_stats()
        assert stats["total_cached_blocks"] == 2
        assert stats["num_nodes"] >= 1

    def test_hit_rate_all_hits(self, cache, block_manager):
        bids = block_manager.allocate(1)
        cache.insert([1, 2, 3, 4], bids)

        cache.match_prefix([1, 2, 3, 4])
        cache.match_prefix([1, 2, 3, 4])

        stats = cache.get_stats()
        assert stats["hit_rate"] == 1.0
        assert stats["total_hits"] == 2
        assert stats["total_misses"] == 0

    def test_hit_rate_mixed(self, cache, block_manager):
        bids = block_manager.allocate(1)
        cache.insert([1, 2, 3, 4], bids)

        cache.match_prefix([1, 2, 3, 4])   # hit
        cache.match_prefix([9, 8, 7, 6])   # miss

        stats = cache.get_stats()
        assert abs(stats["hit_rate"] - 0.5) < 1e-6

    def test_stats_after_evict(self, cache, block_manager):
        bids = block_manager.allocate(2)
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8], bids)

        cache.evict(2)

        stats = cache.get_stats()
        assert stats["total_cached_blocks"] == 0
        assert stats["num_nodes"] == 0

    def test_stats_keys_present(self, cache):
        stats = cache.get_stats()
        for key in ("hit_rate", "total_hits", "total_misses",
                    "total_cached_blocks", "num_nodes"):
            assert key in stats, f"Missing key: {key}"
