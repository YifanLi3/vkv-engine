# Phase 1: Block-based KV Cache Memory Management

> **Est. time**: 10–15 hours | **Difficulty**: ★★★☆☆
> **Prerequisites**: PyTorch tensor ops, basic data structures (linked list, hash map), the OS virtual-memory idea
> **Requires GPU**: No (all tests can run on CPU with `device="cpu"`)

### Naming reference

Class and module names in this project are aligned with nano-vLLM / vLLM so you can read their source later:

| vkv-engine (this project) | nano-vLLM | vLLM | Corresponding OS concept |
|---------------------------|-----------|------|--------------------------|
| `Block` | `Block` | `PhysicalTokenBlock` | Physical page frame |
| `BlockAllocator` | inline in `BlockManager` | `BlockAllocator` | Free-frame list |
| `BlockManager` | `BlockManager` | `BlockSpaceManager` | Physical memory manager |
| `Sequence` | `Sequence` | `Sequence` | Process (with page table) |
| `SequenceStatus` | `SequenceStatus` | `SequenceStatus` | Process state |
| `Scheduler` (Phase 2) | `Scheduler` | `Scheduler` | Process scheduler |
| `ModelRunner` (Phase 2) | `ModelRunner` | `ModelRunner` | — |
| `SamplingParams` | `SamplingParams` | `SamplingParams` | — |

---

## Table of Contents

- [Part 0: Background — why Block-based memory](#part-0-background)
- [Part 1: Understanding KV cache shapes (warm-up)](#part-1-kv-cache-shapes)
- [Part 2: Block — the smallest KV-cache storage unit (core)](#part-2-block)
- [Part 3: BlockAllocator — free-block management (core)](#part-3-block-allocator)
- [Part 4: BlockManager — the pre-allocated memory pool (core)](#part-4-block-manager)
- [Part 5: Sequence — per-request KV management (core)](#part-5-sequence)
- [Part 6: LRUEvictor — what to do when memory runs out (advanced)](#part-6-eviction)
- [Part 7: Swapper — GPU ↔ CPU swap (advanced)](#part-7-swap)
- [Part 8: Defragmentation (Bonus)](#part-8-defragmentation)

---

## How to use this document

1. **Read the background first**: each Part starts with concepts; make sure you understand them before coding.
2. **Fill in the code**: look for `raise NotImplementedError("TODO: ...")` markers in the `.py` files under `vkv/engine/`.
3. **Run the tests**: after finishing each Part, run the corresponding tests:
   ```bash
   # Tests for a single Part
   uv run pytest tests/test_phase1.py -k "part1" -v

   # All Phase 1 tests
   uv run pytest tests/test_phase1.py -v
   ```
4. **Progressive development**: Parts 1–5 are core (must do); Parts 6–8 are advanced/Bonus.

---

<a id="part-0-background"></a>
## Part 0: Background — Why Block-based memory

### Problem: naive KV cache storage

In the simplest LLM inference implementation, each request's KV cache is a
**contiguous, ever-growing tensor**:

```python
# Simplified logic of HuggingFace's DynamicCache
class NaiveKVCache:
    def update(self, layer_idx, new_key, new_value):
        self.key_cache[layer_idx] = torch.cat(
            [self.key_cache[layer_idx], new_key], dim=2
        )
        # Problem: torch.cat allocates new memory + copies old data → fragmentation!
```

As concurrency grows to dozens or hundreds of requests, this scheme leads to:

```
GPU memory (naive approach):
  ├──[Seq A: 2KB]──[gap]──[Seq B: 8KB]──[gap]──[Seq C: 1KB]──[gap]──┤
  Total gap = 10 KB. Even though total free > 10 KB, you cannot allocate a 10 KB
  contiguous block!
```

### Solution: borrow OS virtual memory → PagedAttention

vLLM's PagedAttention applies OS paging to the KV cache:

```
BlockManager's memory pool (pre-allocated):
  [BLK 0][BLK 1][BLK 2][BLK 3][BLK 4][BLK 5][BLK 6][BLK 7] ...

Sequence A's block_table: [0, 2, 3]      ← doesn't need to be contiguous!
Sequence B's block_table: [1, 5, 6]

Each block has a fixed size (e.g. 16 tokens), allocated on demand.
When a request finishes, its blocks are returned to the pool — no fragmentation.
```

nano-vLLM implements this in roughly 200 lines. Our Phase 1 breaks it into 5 steps so you understand each layer.

---

<a id="part-1-kv-cache-shapes"></a>
## Part 1: Understanding KV cache shapes [warm-up]

> **File**: `kv_cache_size_per_token()` in `vkv/engine/block.py`
> **Tests**: `uv run pytest tests/test_phase1.py -k "part1" -v`

### Background: KV cache tensor dimensions

For a single token in a single attention layer:

```
Key shape:   [num_kv_heads, head_dim]
Value shape: [num_kv_heads, head_dim]
```

Whole model: `num_layers × 2(K+V) × num_kv_heads × head_dim × element_size(bytes)`

**GQA note**: Llama 3 uses GQA — number of KV heads (8) is much smaller than Q heads (32).

### Example

```
Llama-3.1-8B: 32 × 2 × 8 × 128 × 2 = 131,072 bytes = 128 KB / token
  1024 tokens = 128 MB,  8192 tokens = 1 GB
```

### Task 1.1: Implement `kv_cache_size_per_token()`
### Task 1.2: Implement `num_blocks_for_tokens()`
### Task 1.3: Implement `kv_block_size_bytes()`

---

<a id="part-2-block"></a>
## Part 2: Block — the smallest KV-cache storage unit [core]

> **File**: `Block` class in `vkv/engine/block.py`
> **Tests**: `uv run pytest tests/test_phase1.py -k "part2" -v`

### Background

In nano-vLLM, `Block` is a very lightweight class:

```python
# nano-vLLM's Block (metadata only)
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1          # for prefix caching
        self.token_ids = []
```

nano-vLLM stores KV tensors in a pre-allocated pool inside `ModelRunner`; the Block is just an index.

Our Part 2 lets each Block hold its own tensor (approach A), so you can see the data reads/writes directly. Part 4 switches to the pre-allocated pool (approach B), matching nano-vLLM's real setup.

### Block internal structure

```
One Block (block_size=16):
┌──────────────────────────────────────────────────────────┐
│ Layer 0: Key   [num_kv_heads, 16, head_dim]              │
│ Layer 0: Value [num_kv_heads, 16, head_dim]              │
│ ...                                                       │
│ Layer N: Key   [num_kv_heads, 16, head_dim]              │
│ Layer N: Value [num_kv_heads, 16, head_dim]              │
└──────────────────────────────────────────────────────────┘
 16 slots; slots 0–12 filled, slots 13–15 waiting for new tokens
```

### Task 2.1: Implement `Block.__init__()` — allocate KV tensor storage
### Task 2.2: Implement `Block.write_slot()` — write a single token's KV
### Task 2.3: Implement `Block.read_slot()` — read KV
### Task 2.4: Implement `Block.clear()` — reset (analogous to nano-vLLM's `Block.reset()`)

---

<a id="part-3-block-allocator"></a>
## Part 3: BlockAllocator — free-block management [core]

> **File**: `vkv/engine/block_allocator.py`
> **Tests**: `uv run pytest tests/test_phase1.py -k "part3" -v`

### Background

In nano-vLLM, free-list management lives inline inside `BlockManager`:

```python
# nano-vLLM's BlockManager.__init__
self.free_block_ids: deque[int] = deque(range(num_blocks))
self.used_block_ids: set[int] = set()
```

We extract it into a standalone `BlockAllocator` (matching vLLM's `BlockAllocator` interface), so you can test allocation logic in isolation and then compose it into `BlockManager` in Part 4.

### Free list operation

```
Initial:  free = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
allocate(3): → [7, 8, 9], free = [0, 1, 2, 3, 4, 5, 6]
allocate(2): → [5, 6],    free = [0, 1, 2, 3, 4]
free([8, 9]): →            free = [0, 1, 2, 3, 4, 8, 9]
```

### Task 3.1: Implement `BlockAllocator.__init__()` — initialize the free list
### Task 3.2: Implement `BlockAllocator.allocate()` — allocate block IDs
### Task 3.3: Implement `BlockAllocator.free()` — return (detect double-free)
### Task 3.4: Implement the `num_free` and `num_used` properties

---

<a id="part-4-block-manager"></a>
## Part 4: BlockManager — the pre-allocated memory pool [core]

> **File**: `vkv/engine/block_manager.py`
> **Tests**: `uv run pytest tests/test_phase1.py -k "part4" -v`

### Background

This is the counterpart of nano-vLLM's `BlockManager`. Key difference:

**nano-vLLM**: KV tensors are pre-allocated inside `ModelRunner`; the BlockManager tracks metadata only.
**vkv-engine**: `BlockManager` owns both metadata and KV tensors — closer to how a vLLM worker does it.

Pre-allocated pool layout:

```python
# Allocated once at startup (nano-vLLM does this in ModelRunner)
for layer in range(num_layers):
    gpu_key_cache[layer] = torch.zeros(
        num_blocks, num_kv_heads, block_size, head_dim,
        dtype=float16, device="cuda"
    )
    gpu_value_cache[layer] = torch.zeros(...)

# Block #5, Layer #3, Slot #7 → Key:
#   gpu_key_cache[3][5, :, 7, :]  → shape [num_kv_heads, head_dim]
```

### Key method mapping vs nano-vLLM

| nano-vLLM | vkv-engine | Description |
|-----------|------------|-------------|
| `BlockManager.allocate(seq)` | `BlockManager.allocate(num_blocks)` | Allocate physical blocks |
| `BlockManager.deallocate(seq)` | `BlockManager.free(block_ids)` | Release (with refcount) |
| `BlockManager.can_allocate(seq)` | `BlockManager.can_allocate(num_blocks)` | Check available capacity |
| `BlockManager.may_append(seq)` | inside `Sequence.append_token()` | Append a block during decode |
| `BlockManager._allocate_block(id)` | `BlockAllocator.allocate()` | Low-level allocation |
| `BlockManager._deallocate_block(id)` | `BlockAllocator.free()` | Low-level release |

### Task 4.1: Implement `BlockManager.__init__()` — pre-allocate the KV tensor pool
### Task 4.2: Implement `BlockManager.allocate()` — allocate + initialize refcount
### Task 4.3: Implement `BlockManager.free()` — decrement refcount, release at 0
### Task 4.4: Implement `BlockManager.write_kv()` / `read_kv()`
### Task 4.5: Implement `BlockManager.gather_kv()` — concatenate KV across blocks
### Task 4.6: Implement `BlockManager.copy_block()` — COW copy

---

<a id="part-5-sequence"></a>
## Part 5: Sequence — per-request KV management [core]

> **File**: `vkv/engine/sequence.py`
> **Tests**: `uv run pytest tests/test_phase1.py -k "part5" -v`

### Background

nano-vLLM's `Sequence` is the central data structure, holding both:
- **Request metadata**: `seq_id`, `status`, `token_ids`, `sampling_params`
- **Block mapping table**: `block_table` — literally the "page table"

```python
# nano-vLLM's Sequence
class Sequence:
    def __init__(self, token_ids, sampling_params):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.block_table = []         # ← this is the "page table"
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
```

Our `Sequence` adds `allocate()`, `append_token()`, `fork()`, and `free()` methods, moving logic that nano-vLLM scattered across `BlockManager` and `Scheduler` onto Sequence itself.

### SequenceStatus state machine

```
               ┌──────────────┐
  new request  │   WAITING    │  nano-vLLM: SequenceStatus.WAITING
               └──────┬───────┘
                      │ scheduler picks it up, allocates blocks
               ┌──────▼───────┐
               │   RUNNING    │  nano-vLLM: SequenceStatus.RUNNING
               └──┬───────┬───┘
       preempted  │       │  finished
               ┌──▼──┐   │
               │SWAP- │   │  nano-vLLM: no swap, just re-queues as WAITING
               │PED   │   │  vkv: KV saved to CPU memory
               └──┬───┘   │
       resumed    │       │
               ┌──▼───────▼───┐
               │   FINISHED   │  nano-vLLM: SequenceStatus.FINISHED
               └──────────────┘
```

### append_token comparison

```python
# nano-vLLM: two steps
# Step 1: Sequence.append_token(token_id)
seq.token_ids.append(token_id)
seq.num_tokens += 1

# Step 2: BlockManager.may_append(seq) — may allocate a new block
if len(seq) % block_size == 1:   # first token of a new block
    block_id = self.free_block_ids[0]
    self._allocate_block(block_id)
    seq.block_table.append(block_id)

# vkv-engine: single step
block_id, slot_idx = seq.append_token(token_id)
# Internally handles new-block allocation and returns the write slot
```

### Task 5.1: Implement `Sequence.__init__()` — initialize request state
### Task 5.2: Implement `Sequence.allocate()` — allocate blocks for prefill
### Task 5.3: Implement `Sequence.append_token()` — append during decode
### Task 5.4: Implement `Sequence.fork()` — COW prefix sharing
### Task 5.5: Implement `Sequence.free()` — release all blocks

---

<a id="part-6-eviction"></a>
## Part 6: LRUEvictor — what to do when memory runs out [advanced]

> **File**: `vkv/engine/evictor.py`
> **Tests**: `uv run pytest tests/test_phase1.py -k "part6" -v`

### Background

nano-vLLM's preemption is very simple:

```python
# nano-vLLM Scheduler.preempt
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)     # simply drop the KV cache
    self.waiting.appendleft(seq)           # re-queue, will re-prefill later
```

This means preempted sequences must **redo their prefill from scratch**, wasting prior compute.

Our LRUEvictor is smarter: it tracks access order and picks the "coldest" sequence to evict. Combined with Part 7's Swapper, an evicted sequence's KV can be saved to CPU rather than thrown away.

### Task 6.1: Implement `LRUEvictor.__init__()` / `add()` / `remove()` / `touch()`
### Task 6.2: Implement `LRUEvictor.evict()` — evict the oldest sequence

---

<a id="part-7-swap"></a>
## Part 7: Swapper — GPU ↔ CPU swap [advanced]

> **File**: `vkv/engine/swapper.py`
> **Tests**: `uv run pytest tests/test_phase1.py -k "part7" -v`

### Background

nano-vLLM has no swap: a preempted sequence's KV cache is simply dropped.

vLLM has a full swap machinery that moves KV cache between GPU ↔ CPU via PCIe. This is one place where vkv-engine goes deeper than nano-vLLM.

### Task 7.1: Implement `Swapper.swap_out()` — GPU → CPU
### Task 7.2: Implement `Swapper.swap_in()` — CPU → GPU

---

<a id="part-8-defragmentation"></a>
## Part 8: Defragmentation [Bonus]

> **File**: `compute_fragmentation()` in `vkv/engine/block_manager.py`
> **Tests**: `uv run pytest tests/test_phase1.py -k "part8" -v`

### Task 8.1: Implement `BlockManager.compute_fragmentation()` — compute internal fragmentation

---

## Running the tests

```bash
# Everything
uv run pytest tests/test_phase1.py -v

# By Part
uv run pytest tests/test_phase1.py -k "part1" -v
uv run pytest tests/test_phase1.py -k "part2" -v
uv run pytest tests/test_phase1.py -k "part3" -v
uv run pytest tests/test_phase1.py -k "part4" -v
uv run pytest tests/test_phase1.py -k "part5" -v
uv run pytest tests/test_phase1.py -k "part6" -v
uv run pytest tests/test_phase1.py -k "part7" -v
uv run pytest tests/test_phase1.py -k "part8" -v

# Single test
uv run pytest tests/test_phase1.py::TestPart4::test_pool_write_read_kv -v
```
