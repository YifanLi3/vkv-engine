"""
Phase 1 Test Suite — aligned with nano-vLLM/vLLM naming

Run all tests:       uv run pytest tests/test_phase1.py -v
Run one Part:        uv run pytest tests/test_phase1.py -k "part1" -v
Run one test:        uv run pytest tests/test_phase1.py::TestPart4::test_allocate_and_free -v
"""

import pytest
import torch

from vkv.config import ModelConfig, CacheConfig, LLAMA_8B, LLAMA_70B, TINY_MODEL


# =============================================================================
# Part 1: KV Cache Size Calculations
# =============================================================================

class TestPart1:
    """Tests for Part 1: kv_cache_size_per_token, num_blocks_for_tokens"""

    def test_kv_size_per_token_llama8b(self):
        from vkv.engine.block import kv_cache_size_per_token
        assert kv_cache_size_per_token(LLAMA_8B) == 131072

    def test_kv_size_per_token_llama70b(self):
        from vkv.engine.block import kv_cache_size_per_token
        assert kv_cache_size_per_token(LLAMA_70B) == 327680

    def test_kv_size_per_token_tiny(self):
        from vkv.engine.block import kv_cache_size_per_token
        assert kv_cache_size_per_token(TINY_MODEL) == 4096

    def test_kv_size_per_token_fp32(self):
        from vkv.engine.block import kv_cache_size_per_token
        config_fp32 = ModelConfig(num_layers=4, num_kv_heads=4, head_dim=64, dtype=torch.float32)
        config_fp16 = ModelConfig(num_layers=4, num_kv_heads=4, head_dim=64, dtype=torch.float16)
        assert kv_cache_size_per_token(config_fp32) == 2 * kv_cache_size_per_token(config_fp16)

    def test_num_blocks_exact(self):
        from vkv.engine.block import num_blocks_for_tokens
        assert num_blocks_for_tokens(32, 16) == 2

    def test_num_blocks_remainder(self):
        from vkv.engine.block import num_blocks_for_tokens
        assert num_blocks_for_tokens(50, 16) == 4

    def test_num_blocks_one_token(self):
        from vkv.engine.block import num_blocks_for_tokens
        assert num_blocks_for_tokens(1, 16) == 1

    def test_num_blocks_zero(self):
        from vkv.engine.block import num_blocks_for_tokens
        assert num_blocks_for_tokens(0, 16) == 0

    def test_kv_block_size_bytes(self):
        from vkv.engine.block import kv_block_size_bytes
        assert kv_block_size_bytes(LLAMA_8B, 16) == 131072 * 16


# =============================================================================
# Part 2: Block Data Structure
# =============================================================================

class TestPart2:
    """Tests for Part 2: Block init, write_slot, read_slot, clear"""

    @pytest.fixture
    def block(self):
        from vkv.engine.block import Block
        return Block(
            block_id=0, num_layers=2, num_kv_heads=4,
            head_dim=64, block_size=16, dtype=torch.float16, device="cpu",
        )

    def test_block_init_shapes(self, block):
        assert block.key_cache.shape == (2, 4, 16, 64)
        assert block.value_cache.shape == (2, 4, 16, 64)

    def test_block_init_dtype(self, block):
        assert block.key_cache.dtype == torch.float16

    def test_block_init_zeros(self, block):
        assert block.key_cache.sum() == 0

    def test_block_init_metadata(self, block):
        assert block.num_filled == 0
        assert block.ref_count == 1
        assert block.last_access == 0.0
        assert block.block_id == 0

    def test_block_properties(self, block):
        assert not block.is_full
        assert block.num_empty_slots == 16

    def test_write_and_read_slot(self, block):
        key = torch.randn(4, 64, dtype=torch.float16)
        value = torch.randn(4, 64, dtype=torch.float16)
        block.write_slot(layer_idx=0, slot_idx=0, key=key, value=value)
        rk, rv = block.read_slot(layer_idx=0, slot_idx=0)
        assert torch.equal(rk, key)
        assert torch.equal(rv, value)

    def test_write_multiple_slots(self, block):
        k0 = torch.randn(4, 64, dtype=torch.float16)
        v0 = torch.randn(4, 64, dtype=torch.float16)
        k1 = torch.randn(4, 64, dtype=torch.float16)
        v1 = torch.randn(4, 64, dtype=torch.float16)
        block.write_slot(layer_idx=0, slot_idx=0, key=k0, value=v0)
        block.write_slot(layer_idx=1, slot_idx=5, key=k1, value=v1)
        rk0, rv0 = block.read_slot(layer_idx=0, slot_idx=0)
        rk1, rv1 = block.read_slot(layer_idx=1, slot_idx=5)
        assert torch.equal(rk0, k0) and torch.equal(rk1, k1)

    def test_write_slot_bounds_check(self, block):
        k = torch.randn(4, 64, dtype=torch.float16)
        v = torch.randn(4, 64, dtype=torch.float16)
        with pytest.raises((IndexError, ValueError)):
            block.write_slot(layer_idx=0, slot_idx=16, key=k, value=v)

    def test_clear(self, block):
        block.num_filled = 10
        block.clear()
        assert block.num_filled == 0

    def test_memory_bytes(self, block):
        expected = 2 * 4 * 16 * 64 * 2 * 2
        assert block.memory_bytes() == expected


# =============================================================================
# Part 3: BlockAllocator
# =============================================================================

class TestPart3:
    """Tests for Part 3: BlockAllocator"""

    def test_init(self):
        from vkv.engine.block_allocator import BlockAllocator
        alloc = BlockAllocator(num_blocks=100)
        assert alloc.num_free == 100
        assert alloc.num_used == 0
        assert alloc.total == 100

    def test_allocate_basic(self):
        from vkv.engine.block_allocator import BlockAllocator
        alloc = BlockAllocator(num_blocks=100)
        ids = alloc.allocate(3)
        assert len(ids) == 3
        assert alloc.num_free == 97

    def test_allocate_unique_ids(self):
        from vkv.engine.block_allocator import BlockAllocator
        alloc = BlockAllocator(num_blocks=100)
        ids = alloc.allocate(50)
        assert len(set(ids)) == 50

    def test_allocate_valid_range(self):
        from vkv.engine.block_allocator import BlockAllocator
        alloc = BlockAllocator(num_blocks=100)
        ids = alloc.allocate(100)
        for bid in ids:
            assert 0 <= bid < 100

    def test_allocate_all(self):
        from vkv.engine.block_allocator import BlockAllocator
        alloc = BlockAllocator(num_blocks=10)
        alloc.allocate(10)
        assert alloc.num_free == 0

    def test_allocate_insufficient(self):
        from vkv.engine.block_allocator import BlockAllocator
        alloc = BlockAllocator(num_blocks=5)
        with pytest.raises(ValueError):
            alloc.allocate(6)

    def test_free_basic(self):
        from vkv.engine.block_allocator import BlockAllocator
        alloc = BlockAllocator(num_blocks=100)
        ids = alloc.allocate(5)
        alloc.free(ids)
        assert alloc.num_free == 100

    def test_free_and_reallocate(self):
        from vkv.engine.block_allocator import BlockAllocator
        alloc = BlockAllocator(num_blocks=5)
        ids = alloc.allocate(5)
        alloc.free(ids[:3])
        new_ids = alloc.allocate(3)
        assert len(new_ids) == 3
        assert alloc.num_free == 0

    def test_double_free(self):
        from vkv.engine.block_allocator import BlockAllocator
        alloc = BlockAllocator(num_blocks=10)
        ids = alloc.allocate(3)
        alloc.free(ids)
        with pytest.raises(ValueError):
            alloc.free(ids)

    def test_has_free(self):
        from vkv.engine.block_allocator import BlockAllocator
        alloc = BlockAllocator(num_blocks=5)
        assert alloc.has_free(5)
        assert not alloc.has_free(6)
        alloc.allocate(3)
        assert alloc.has_free(2)
        assert not alloc.has_free(3)

    def test_stress_allocate_free_cycle(self):
        from vkv.engine.block_allocator import BlockAllocator
        import random
        alloc = BlockAllocator(num_blocks=1000)
        allocated = []
        for _ in range(500):
            n = random.randint(1, min(10, alloc.num_free)) if alloc.num_free > 0 else 0
            if n > 0:
                allocated.append(alloc.allocate(n))
            if allocated and random.random() > 0.5:
                alloc.free(allocated.pop(random.randint(0, len(allocated) - 1)))
        for ids in allocated:
            alloc.free(ids)
        assert alloc.num_free == 1000


# =============================================================================
# Part 4: BlockManager
# =============================================================================

class TestPart4:
    """Tests for Part 4: BlockManager"""

    @pytest.fixture
    def mgr(self):
        from vkv.engine.block_manager import BlockManager
        model_cfg = TINY_MODEL
        cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=100, num_cpu_blocks=50)
        return BlockManager(model_cfg, cache_cfg, device="cpu")

    def test_pool_init_shapes(self, mgr):
        assert len(mgr.gpu_key_cache) == 4
        for layer in range(4):
            assert mgr.gpu_key_cache[layer].shape == (100, 4, 16, 64)

    def test_pool_init_cpu_shapes(self, mgr):
        assert len(mgr.cpu_key_cache) == 4
        for layer in range(4):
            assert mgr.cpu_key_cache[layer].shape == (50, 4, 16, 64)

    def test_allocate_basic(self, mgr):
        ids = mgr.allocate(5)
        assert len(ids) == 5
        assert mgr.stats.used_blocks == 5

    def test_free_basic(self, mgr):
        ids = mgr.allocate(5)
        mgr.free(ids)
        assert mgr.stats.used_blocks == 0

    def test_ref_count(self, mgr):
        ids = mgr.allocate(3)
        mgr.inc_ref(ids[0])
        mgr.free(ids)
        assert mgr.get_ref_count(ids[0]) == 1
        assert mgr.stats.used_blocks == 1
        mgr.free([ids[0]])
        assert mgr.stats.used_blocks == 0

    def test_write_read_kv(self, mgr):
        ids = mgr.allocate(1)
        key = torch.randn(4, 64, dtype=torch.float16)
        value = torch.randn(4, 64, dtype=torch.float16)
        mgr.write_kv(ids[0], layer_idx=0, slot_idx=0, key=key, value=value)
        rk, rv = mgr.read_kv(ids[0], layer_idx=0, slot_idx=0)
        assert torch.equal(rk, key)
        assert torch.equal(rv, value)

    def test_write_isolation(self, mgr):
        ids = mgr.allocate(2)
        k0 = torch.randn(4, 64, dtype=torch.float16)
        v0 = torch.randn(4, 64, dtype=torch.float16)
        k1 = torch.randn(4, 64, dtype=torch.float16)
        v1 = torch.randn(4, 64, dtype=torch.float16)
        mgr.write_kv(ids[0], 0, 0, k0, v0)
        mgr.write_kv(ids[1], 1, 5, k1, v1)
        rk0, _ = mgr.read_kv(ids[0], 0, 0)
        rk1, _ = mgr.read_kv(ids[1], 1, 5)
        assert torch.equal(rk0, k0) and torch.equal(rk1, k1)

    def test_gather_kv(self, mgr):
        ids = mgr.allocate(3)
        for i, bid in enumerate(ids):
            for slot in range(16):
                k = torch.full((4, 64), float(i * 16 + slot), dtype=torch.float16)
                v = torch.full((4, 64), float(i * 16 + slot + 0.5), dtype=torch.float16)
                mgr.write_kv(bid, layer_idx=0, slot_idx=slot, key=k, value=v)
        keys, values = mgr.gather_kv(block_ids=ids, num_tokens=40, layer_idx=0)
        assert keys.shape == (1, 4, 40, 64)
        assert torch.all(keys[0, 0, 0, :] == 0.0)
        assert torch.all(keys[0, 0, 16, :] == 16.0)
        assert torch.all(keys[0, 0, 39, :] == 39.0)

    def test_gather_kv_full_blocks(self, mgr):
        ids = mgr.allocate(2)
        for i, bid in enumerate(ids):
            for slot in range(16):
                k = torch.ones(4, 64, dtype=torch.float16) * (i * 16 + slot)
                v = torch.ones(4, 64, dtype=torch.float16)
                mgr.write_kv(bid, 0, slot, k, v)
        keys, _ = mgr.gather_kv(ids, num_tokens=32, layer_idx=0)
        assert keys.shape == (1, 4, 32, 64)

    def test_copy_block(self, mgr):
        ids = mgr.allocate(2)
        key = torch.randn(4, 64, dtype=torch.float16)
        value = torch.randn(4, 64, dtype=torch.float16)
        mgr.write_kv(ids[0], 0, 0, key, value)
        mgr.copy_block(ids[0], ids[1])
        rk, rv = mgr.read_kv(ids[1], 0, 0)
        assert torch.equal(rk, key)


# =============================================================================
# Part 5: Sequence
# =============================================================================

class TestPart5:
    """Tests for Part 5: Sequence (was BlockTable)"""

    @pytest.fixture
    def mgr(self):
        from vkv.engine.block_manager import BlockManager
        model_cfg = TINY_MODEL
        cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=100, num_cpu_blocks=50)
        return BlockManager(model_cfg, cache_cfg, device="cpu")

    def test_sequence_init(self, mgr):
        from vkv.engine.sequence import Sequence, SequenceStatus
        seq = Sequence(token_ids=[1, 2, 3], block_manager=mgr)
        assert seq.num_tokens == 3
        assert seq.num_prompt_tokens == 3
        assert seq.status == SequenceStatus.WAITING
        assert seq.block_table == []
        assert len(seq) == 3

    def test_allocate(self, mgr):
        from vkv.engine.sequence import Sequence, SequenceStatus
        seq = Sequence(token_ids=list(range(50)), block_manager=mgr)
        seq.allocate()
        assert len(seq.block_table) == 4  # ceil(50/16) = 4
        assert seq.status == SequenceStatus.RUNNING

    def test_allocate_exact(self, mgr):
        from vkv.engine.sequence import Sequence
        seq = Sequence(token_ids=list(range(32)), block_manager=mgr)
        seq.allocate()
        assert len(seq.block_table) == 2

    def test_append_token_within_block(self, mgr):
        from vkv.engine.sequence import Sequence
        seq = Sequence(token_ids=list(range(10)), block_manager=mgr)
        seq.allocate()
        initial_blocks = len(seq.block_table)
        block_id, slot_idx = seq.append_token(token_id=999)
        assert seq.num_tokens == 11
        assert len(seq.block_table) == initial_blocks
        assert slot_idx == 10

    def test_append_token_new_block(self, mgr):
        from vkv.engine.sequence import Sequence
        seq = Sequence(token_ids=list(range(16)), block_manager=mgr)
        seq.allocate()
        block_id, slot_idx = seq.append_token(token_id=999)
        assert seq.num_tokens == 17
        assert len(seq.block_table) == 2
        assert slot_idx == 0

    def test_append_many_tokens(self, mgr):
        from vkv.engine.sequence import Sequence
        seq = Sequence(token_ids=[0], block_manager=mgr)
        seq.allocate()
        for i in range(47):
            seq.append_token(token_id=i + 1)
        assert seq.num_tokens == 48
        assert len(seq.block_table) == 3

    def test_fork(self, mgr):
        from vkv.engine.sequence import Sequence
        original = Sequence(token_ids=list(range(50)), block_manager=mgr)
        original.allocate()
        forked = original.fork()
        assert forked.num_tokens == original.num_tokens
        assert forked.block_table == original.block_table
        assert forked is not original
        for bid in original.block_table:
            assert mgr.get_ref_count(bid) == 2

    def test_fork_independent_append(self, mgr):
        from vkv.engine.sequence import Sequence
        original = Sequence(token_ids=list(range(16)), block_manager=mgr)
        original.allocate()
        forked = original.fork()
        forked.append_token(token_id=999)
        assert original.num_tokens == 16
        assert forked.num_tokens == 17
        assert len(original.block_table) == 1
        assert len(forked.block_table) == 2

    def test_free(self, mgr):
        from vkv.engine.sequence import Sequence
        initial_free = mgr.stats.free_blocks
        seq = Sequence(token_ids=list(range(50)), block_manager=mgr)
        seq.allocate()
        assert mgr.stats.free_blocks == initial_free - 4
        seq.free()
        assert mgr.stats.free_blocks == initial_free
        assert seq.block_table == []

    def test_free_with_shared_blocks(self, mgr):
        from vkv.engine.sequence import Sequence
        seq_a = Sequence(token_ids=list(range(32)), block_manager=mgr)
        seq_a.allocate()
        seq_b = seq_a.fork()
        seq_a.free()
        for bid in seq_b.block_table:
            assert mgr.get_ref_count(bid) == 1
        seq_b.free()
        assert mgr.stats.used_blocks == 0


# =============================================================================
# Part 6: LRU Evictor
# =============================================================================

class TestPart6:
    """Tests for Part 6: LRUEvictor"""

    def test_add_and_len(self):
        from vkv.engine.evictor import LRUEvictor
        ev = LRUEvictor()
        ev.add("seq_0", [0, 1, 2])
        ev.add("seq_1", [3, 4])
        assert len(ev) == 2

    def test_remove(self):
        from vkv.engine.evictor import LRUEvictor
        ev = LRUEvictor()
        ev.add("seq_0", [0, 1, 2])
        ev.remove("seq_0")
        assert len(ev) == 0

    def test_remove_nonexistent(self):
        from vkv.engine.evictor import LRUEvictor
        ev = LRUEvictor()
        ev.remove("no_such")

    def test_evict_lru_order(self):
        from vkv.engine.evictor import LRUEvictor
        ev = LRUEvictor()
        ev.add("seq_0", [0, 1])
        ev.add("seq_1", [2, 3])
        ev.add("seq_2", [4, 5])
        victims = ev.evict(2)
        assert victims[0][0] == "seq_0"

    def test_evict_multiple(self):
        from vkv.engine.evictor import LRUEvictor
        ev = LRUEvictor()
        ev.add("seq_0", [0])
        ev.add("seq_1", [1, 2])
        ev.add("seq_2", [3, 4, 5])
        victims = ev.evict(4)
        total_freed = sum(len(v[1]) for v in victims)
        assert total_freed >= 4

    def test_touch_changes_order(self):
        from vkv.engine.evictor import LRUEvictor
        ev = LRUEvictor()
        ev.add("seq_0", [0, 1])
        ev.add("seq_1", [2, 3])
        ev.add("seq_2", [4, 5])
        ev.touch("seq_0")
        victims = ev.evict(2)
        assert victims[0][0] == "seq_1"

    def test_evict_insufficient(self):
        from vkv.engine.evictor import LRUEvictor
        ev = LRUEvictor()
        ev.add("seq_0", [0])
        with pytest.raises(ValueError):
            ev.evict(5)

    def test_num_evictable_blocks(self):
        from vkv.engine.evictor import LRUEvictor
        ev = LRUEvictor()
        ev.add("seq_0", [0, 1, 2])
        ev.add("seq_1", [3, 4])
        assert ev.num_evictable_blocks == 5

    def test_update_blocks(self):
        from vkv.engine.evictor import LRUEvictor
        ev = LRUEvictor()
        ev.add("seq_0", [0])
        ev.update_blocks("seq_0", [0, 1, 2])
        assert ev.num_evictable_blocks == 3


# =============================================================================
# Part 7: Swapper
# =============================================================================

class TestPart7:
    """Tests for Part 7: GPU ↔ CPU Swap"""

    @pytest.fixture
    def mgr(self):
        from vkv.engine.block_manager import BlockManager
        model_cfg = TINY_MODEL
        cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=50, num_cpu_blocks=50)
        return BlockManager(model_cfg, cache_cfg, device="cpu")

    def test_swap_out_basic(self, mgr):
        from vkv.engine.swapper import Swapper
        swapper = Swapper(mgr)
        gpu_ids = mgr.allocate(2)
        key = torch.randn(4, 64, dtype=torch.float16)
        value = torch.randn(4, 64, dtype=torch.float16)
        mgr.write_kv(gpu_ids[0], 0, 0, key, value)
        initial_free = mgr.gpu_allocator.num_free
        mapping = swapper.swap_out(gpu_ids)
        assert len(mapping) == 2
        assert mgr.gpu_allocator.num_free == initial_free + 2

    def test_swap_out_preserves_data(self, mgr):
        from vkv.engine.swapper import Swapper
        swapper = Swapper(mgr)
        gpu_ids = mgr.allocate(1)
        key = torch.randn(4, 64, dtype=torch.float16)
        value = torch.randn(4, 64, dtype=torch.float16)
        mgr.write_kv(gpu_ids[0], 0, 0, key, value)
        mapping = swapper.swap_out(gpu_ids)
        cpu_id = mapping[gpu_ids[0]]
        assert torch.equal(mgr.cpu_key_cache[0][cpu_id, :, 0, :], key)
        assert torch.equal(mgr.cpu_value_cache[0][cpu_id, :, 0, :], value)

    def test_swap_in_basic(self, mgr):
        from vkv.engine.swapper import Swapper
        swapper = Swapper(mgr)
        gpu_ids = mgr.allocate(1)
        key = torch.randn(4, 64, dtype=torch.float16)
        value = torch.randn(4, 64, dtype=torch.float16)
        mgr.write_kv(gpu_ids[0], 0, 0, key, value)
        out_map = swapper.swap_out(gpu_ids)
        in_map = swapper.swap_in([out_map[gpu_ids[0]]])
        new_gpu_id = in_map[out_map[gpu_ids[0]]]
        rk, rv = mgr.read_kv(new_gpu_id, 0, 0)
        assert torch.equal(rk, key)

    def test_swap_roundtrip_all_layers(self, mgr):
        from vkv.engine.swapper import Swapper
        swapper = Swapper(mgr)
        gpu_ids = mgr.allocate(1)
        bid = gpu_ids[0]
        expected = {}
        for layer in range(mgr.num_layers):
            k = torch.randn(4, 64, dtype=torch.float16)
            v = torch.randn(4, 64, dtype=torch.float16)
            mgr.write_kv(bid, layer, 0, k, v)
            expected[layer] = (k, v)
        out_map = swapper.swap_out([bid])
        in_map = swapper.swap_in([out_map[bid]])
        new_bid = in_map[out_map[bid]]
        for layer in range(mgr.num_layers):
            rk, rv = mgr.read_kv(new_bid, layer, 0)
            assert torch.equal(rk, expected[layer][0])
            assert torch.equal(rv, expected[layer][1])


# =============================================================================
# Part 8: Defragmentation (Bonus)
# =============================================================================

class TestPart8:
    """Tests for Part 8: Fragmentation measurement (Bonus)"""

    def test_fragmentation_no_waste(self):
        from vkv.engine.block_manager import BlockManager
        from vkv.engine.sequence import Sequence
        mgr = BlockManager(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=100, num_cpu_blocks=50), device="cpu")
        seq = Sequence(token_ids=list(range(32)), block_manager=mgr)
        seq.allocate()
        frag = mgr.compute_fragmentation([seq])
        assert frag == 0.0

    def test_fragmentation_with_waste(self):
        from vkv.engine.block_manager import BlockManager
        from vkv.engine.sequence import Sequence
        mgr = BlockManager(TINY_MODEL, CacheConfig(block_size=16, num_gpu_blocks=100, num_cpu_blocks=50), device="cpu")
        seq = Sequence(token_ids=list(range(17)), block_manager=mgr)
        seq.allocate()
        frag = mgr.compute_fragmentation([seq])
        assert abs(frag - 15 / 32) < 1e-6
