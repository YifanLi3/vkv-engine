# Phase 4: Prefix Caching — Radix Tree + Copy-on-Write

> **Est. time**: 10–15 hours | **Difficulty**: ★★★★★
> **Prerequisites**: Phase 1 done (BlockManager, Sequence, ref counting)
> **Requires GPU**: No (all tests run on CPU)

---

## Table of Contents

- [Part 0: Background — What is prefix caching and why it matters](#part-0-background)
- [Part 1: RadixNode — understand the data structure](#part-1-radix-node)
- [Part 2: match_prefix — longest prefix lookup](#part-2-match-prefix)
- [Part 3: insert — adding sequences to the cache](#part-3-insert)
- [Part 4: evict — LRU block eviction](#part-4-evict)
- [Part 5: copy_on_write — COW for shared blocks](#part-5-cow)
- [Part 6: get_stats — cache statistics](#part-6-stats)

---

<a id="part-0-background"></a>
## Part 0: Background — What is prefix caching and why it matters

### The problem: repeated prefill is expensive

In production LLM serving, many requests share a common prefix:

```
System prompt (1000 tokens) → every single request
User A: [system_prompt] + "What is Python?"
User B: [system_prompt] + "Explain neural networks"
User C: [system_prompt] + "Write a haiku"
```

Without prefix caching, every request runs prefill on the full 1000-token
system prompt. This is pure wasted compute — the KV for those 1000 tokens
is identical every time.

### The solution: cache and reuse KV blocks

```
First request: prefill 1000 tokens → store KV blocks in cache
Second request: match_prefix → find 1000 cached tokens → skip prefill!
                               only prefill the new tokens
```

Savings:
```
Without prefix cache: 100 requests × 1000 tokens = 100,000 tokens prefilled
With prefix cache:    1 prefill + 99 cache hits   = 1,000  tokens prefilled
                                                    99× speedup on prefill
```

### How does the Radix Tree help?

A simple hash map can cache exact full prefixes. But a Radix Tree can match
any common prefix, even between different sequences:

```
Sequence A: [sys_prompt] + chat_history_A + new_question_A
Sequence B: [sys_prompt] + chat_history_A + new_question_B
             ─────────────────────────────── shared prefix!
```

The radix tree finds this shared prefix in O(prefix_length) time and reuses
all the corresponding blocks.

### Copy-on-Write (COW)

When a cached block is shared between the cache and an active sequence, we
can't let the sequence write new tokens into it — that would corrupt the cache.

COW solves this: before the sequence writes to a shared block, we allocate a
fresh copy. The sequence writes to the copy; the cache keeps the original.

```
shared_block (ref=2) ──→ [copy] ──→ new_block (ref=1, write here)
                     └──→ [original stays in cache, ref decremented to 1]
```

---

<a id="part-1-radix-node"></a>
## Part 1: RadixNode — understand the data structure [warm-up]

> **File**: `RadixNode` in `vkv/engine/prefix_cache.py`
> **Tests**: `uv run pytest tests/test_phase4.py -k "part1" -v`

### What to read

`RadixNode` is already implemented. Read the class carefully and understand:

- `token_ids`: the token segment this node represents (NOT the full path, just this node's segment)
- `block_ids`: physical block IDs for the KV data of `token_ids`
- `children`: `Dict[int, RadixNode]` — keyed by the **first token** of the child's segment
- `parent`: reference to parent (for eviction — we need to remove ourselves from parent.children)
- `last_access`: monotonic timestamp, updated on cache hits (drives LRU eviction)
- `ref_count`: number of active sequences currently using this node's blocks (do NOT evict if > 0)

### Tree example (block_size = 2)

```
Three sequences cached:
  A: tokens [1,2,3,4]     → blocks [10, 11]
  B: tokens [1,2,5,6]     → blocks [10, 12]   ← shares block 10 with A!
  C: tokens [7,8]         → blocks [13]

Tree structure:
  root (tokens=[], blocks=[])
    ├─ key=1 → node(tokens=[1,2], blocks=[10])   ← shared prefix node
    │            ├─ key=3 → node(tokens=[3,4], blocks=[11])
    │            └─ key=5 → node(tokens=[5,6], blocks=[12])
    └─ key=7 → node(tokens=[7,8], blocks=[13])
```

### Tasks

**Task 1.1**: Trace through the example above. For each node, write down:
- `token_ids`, `block_ids`, `is_leaf`, `num_tokens`, `num_blocks`

**Task 1.2**: Answer these questions:
- Why is the key in `children` the **first** token of the child's `token_ids`?
- Why does the shared prefix `[1,2]` appear as its own node instead of being
  duplicated in both the `[3,4]` and `[5,6]` children?
- When we evict node `[3,4]`, does block 10 get freed? Why or why not?

---

<a id="part-2-match-prefix"></a>
## Part 2: match_prefix — longest prefix lookup [core]

> **File**: `PrefixCache.match_prefix()` in `vkv/engine/prefix_cache.py`
> **Tests**: `uv run pytest tests/test_phase4.py -k "part2" -v`

### What it does

Given a new request's `token_ids`, find the longest prefix already cached in
the tree and return the corresponding physical block IDs.

```
Cache contains: [1,2,3,4] → blocks [10, 11]   (block_size=2)
New request:    [1,2,3,4,5,6]
→ match_prefix returns: ([10, 11], 4)
  meaning: reuse blocks 10 and 11, skip prefill for the first 4 tokens
```

### Algorithm

```python
matched_blocks = []
cursor = 0          # index into token_ids
node = self.root

while cursor < len(token_ids):
    key = token_ids[cursor]
    if key not in node.children:
        break                        # no further match possible
    child = node.children[key]
    common = self._match_len(child.token_ids, token_ids[cursor:])

    if common == child.num_tokens:   # full node matched
        matched_blocks.extend(child.block_ids)
        cursor += child.num_tokens
        child.last_access = time.monotonic()
        node = child
    else:
        break                        # partial match — stop here

# update self._hits or self._misses
# return matched_blocks, cursor
```

### Task 2.1: Implement `match_prefix()`

Key points:
- Use `self._match_len()` (provided) to count matching tokens
- Update `child.last_access` on every matched node (for LRU)
- Update `self._hits` if any blocks matched, else `self._misses`
- Return `(matched_blocks, cursor)` — cursor is always a multiple of block_size

### Task 2.2: Trace through this example by hand

```
block_size = 2
Cache:
  root → node([1,2], [10])
            └─ node([3,4], [11])

Query: match_prefix([1,2,3,4,5,6])

Step 1: cursor=0, key=1, find child([1,2],[10])
        common = _match_len([1,2], [1,2,3,4,5,6]) = ?
        full match? → extend matched_blocks, cursor → ?

Step 2: cursor=2, key=3, find child([3,4],[11])
        common = _match_len([3,4], [3,4,5,6]) = ?
        full match? → extend matched_blocks, cursor → ?

Step 3: cursor=4, key=5, no child → break

Result: matched_blocks=?, cursor=?
```

---

<a id="part-3-insert"></a>
## Part 3: insert — adding sequences to the cache [core]

> **File**: `PrefixCache.insert()` in `vkv/engine/prefix_cache.py`
> **Tests**: `uv run pytest tests/test_phase4.py -k "part3" -v`

### What it does

After prefill, store the computed token→block mapping in the radix tree so
future requests with the same prefix can reuse the blocks.

### Alignment rule

Only store **complete** blocks. If `len(token_ids) = 5` and `block_size = 2`,
only cache the first 4 tokens (2 complete blocks). The 5th token is still
being decoded and its block may grow.

```python
num_complete = len(token_ids) // block_size
tokens_to_cache = token_ids[:num_complete * block_size]
blocks_to_cache = block_ids[:num_complete]
```

### Three cases during insertion

**Case 1: No matching child — create a new leaf**

```
Existing: root → node([1,2], [10])
Insert:   [7,8,9,10] → blocks [13, 14]

At root, key=7, no child → create node([7,8,9,10], [13,14]) under root
Call block_manager.inc_ref(13), block_manager.inc_ref(14)
```

**Case 2: Full match — descend and continue**

```
Existing: root → node([1,2], [10])
Insert:   [1,2,3,4] → blocks [10, 11]

At root, key=1, child([1,2],[10]), common=2 == child.num_tokens
→ descend into child, cursor+=2, block_cursor+=1
→ at child, key=3, no child → create node([3,4],[11]) under child
→ inc_ref(11)
```

**Case 3: Partial match — split the existing child**

```
Existing: root → node([1,2,3,4], [10, 11])
Insert:   [1,2,5,6] → blocks [10, 12]

At root, key=1, child([1,2,3,4],[10,11])
common = _match_len([1,2,3,4], [1,2,5,6]) = 2  (common_blocks = 1)
2 < child.num_tokens (4) → SPLIT

Create mid_node:
  token_ids = child.token_ids[:2] = [1,2]
  block_ids = child.block_ids[:1] = [10]

Create suffix_node (old child tail):
  token_ids = child.token_ids[2:] = [3,4]
  block_ids = child.block_ids[1:] = [11]

Create new_node (new tokens):
  token_ids = [5,6]
  block_ids = [12]
  inc_ref(12)

Connect:
  root.children[1] = mid_node
  mid_node.children[3] = suffix_node
  mid_node.children[5] = new_node
```

Note: `mid_node` and `suffix_node` share the same physical blocks as the
original child — **no extra `inc_ref` needed** for them. Only the newly
inserted leaf (`new_node`) needs `inc_ref`.

### ⚠️ Critical rule: always split at a BLOCK boundary, never mid-block

In the example above, `common = 2` happens to divide evenly into `block_size
= 2`. But token-level matching can produce a `common` that is **not** a
multiple of `block_size` — and in that case you must round DOWN to the
nearest block boundary before splitting.

**Why:** a block is the smallest cacheable/shareable unit. A block is only
truly reusable if *every* token inside it matches — one matching token
followed by a differing token still makes it a different physical block.

```
block_size = 2
Existing: root → node([1,2,3,4], blocks=[10, 11])
Insert:   [1,2,3,5] → blocks=[10, 12]

common_raw = _match_len([1,2,3,4], [1,2,3,5]) = 3   # tokens 1,2,3 all match!

BUT: block 11 covers tokens [3,4]; the new sequence's block for tokens [3,5]
     is a DIFFERENT block (id 12) — token 4 vs 5 differ, so the whole block
     differs, even though token 3 alone matched.

→ round down to the block boundary:
     common_blocks = common_raw // block_size = 3 // 2 = 1
     common        = common_blocks * block_size = 2      ← use THIS

So the split only shares [1,2] (1 block), NOT [1,2,3] (1.5 blocks):
  mid_node:    token_ids=[1,2],   block_ids=[10]
  suffix_node: token_ids=[3,4],   block_ids=[11]
  new_node:    token_ids=[3,5],   block_ids=[12]   ← inc_ref(12)
```

**Rule of thumb:** always compute `common_blocks = common_raw // block_size`
first, then derive `common = common_blocks * block_size`. Use this aligned
`common` for ALL slicing (`token_ids[:common]`, `block_ids[:common_blocks]`).
Never slice `token_ids` using the raw, unaligned match length — otherwise a
node's `token_ids` length and `block_ids` length become inconsistent
(`len(token_ids)` must always equal `len(block_ids) * block_size`).

### Task 3.1: Implement `insert()`

### Task 3.2: Insert both [1,2,3,4] and [1,2,5,6] step by step

Draw the tree after each insert. Verify the block ref counts in BlockManager.

---

<a id="part-4-evict"></a>
## Part 4: evict — LRU block eviction [core]

> **File**: `PrefixCache.evict()` in `vkv/engine/prefix_cache.py`
> **Tests**: `uv run pytest tests/test_phase4.py -k "part4" -v`

### What it does

When GPU memory is low, free cached blocks to make room for new requests.
Evict the **least recently used** leaf node first. Never evict a node that
is currently in use (ref_count > 0).

### Why only leaf nodes?

Evicting an internal node would disconnect its subtree — all children would
become unreachable, wasting memory. We always evict leaves first.

After evicting a leaf, its parent may become a new leaf. If that parent also
has `ref_count == 0`, it can be evicted in the next iteration.

### Algorithm

```python
freed = 0
while freed < num_blocks:
    # Collect all eviction candidates
    candidates = []
    self._collect_leaves(self.root, candidates)
    candidates = [n for n in candidates if n.ref_count == 0]

    if not candidates:
        break  # nothing left to evict

    # Pick LRU: oldest last_access
    victim = min(candidates, key=lambda n: n.last_access)

    # Free the blocks
    self.block_manager.free(victim.block_ids)
    freed += victim.num_blocks

    # Detach from parent
    first_token = victim.token_ids[0]
    del victim.parent.children[first_token]

return freed
```

### Task 4.1: Implement `evict()`

### Task 4.2: Answer these questions

- What happens to block 10 (the shared prefix block) if both children
  `[3,4]` and `[5,6]` are evicted? When does block 10 get freed?
- Why do we re-collect leaves on every loop iteration instead of sorting
  once at the start?

---

<a id="part-5-cow"></a>
## Part 5: copy_on_write — COW for shared blocks [core]

> **File**: `PrefixCache.copy_on_write()` in `vkv/engine/prefix_cache.py`
> **Tests**: `uv run pytest tests/test_phase4.py -k "part5" -v`

### What it does

When a sequence needs to write new KV data into a block that is shared with
the prefix cache (ref_count > 1), we must copy the block first to avoid
corrupting the cached data.

```
Before COW:
  block_id=10, ref_count=2  (shared by prefix cache + active sequence)

After COW:
  block_id=10, ref_count=1  (prefix cache only)
  new_block_id=15, ref_count=1  (active sequence private copy)

The sequence then uses new_block_id=15 for its writes.
```

### Steps

```python
new_block_ids = self.block_manager.allocate(1)       # allocate fresh block
new_block_id = new_block_ids[0]
self.block_manager.copy_block(block_id, new_block_id) # copy KV data
self.block_manager.free([block_id])                   # decrement ref on original
return new_block_id
```

### Task 5.1: Implement `copy_on_write()`

### Task 5.2: Integrate COW into a decode loop

After implementing COW, write a short test scenario:

```python
# 1. Prefill sequence A: tokens [1,2,3,4], blocks [10, 11]
# 2. cache.insert([1,2,3,4], [10, 11])
# 3. Sequence A continues decoding — wants to write token 5 into slot 0 of block 11
# 4. block_manager.get_ref_count(11) == 2 → must COW
# 5. new_block = cache.copy_on_write(11)
# 6. Sequence A now uses new_block instead of 11
# 7. block_manager.get_ref_count(11) == 1 (cache still holds it)
# 8. block_manager.get_ref_count(new_block) == 1
```

---

<a id="part-6-stats"></a>
## Part 6: get_stats — cache statistics [advanced]

> **File**: `PrefixCache.get_stats()` in `vkv/engine/prefix_cache.py`
> **Tests**: `uv run pytest tests/test_phase4.py -k "part6" -v`

### What to return

```python
{
    "hit_rate":            float,   # hits / (hits + misses), 0.0 if no queries
    "total_hits":          int,
    "total_misses":        int,
    "total_cached_blocks": int,     # blocks currently stored in the tree
    "num_nodes":           int,     # tree nodes excluding root
}
```

### Task 6.1: Implement `get_stats()`

Use `self._count_nodes_and_blocks(self.root)` (provided) for tree totals.

### Task 6.2: Run the benchmark

After implementing all parts, run the example script:

```bash
uv run examples/run_prefix_cache.py
```

You should see output similar to:
```
=== Prefix Cache Demo ===
Sequences: 10   System prompt: 500 tokens   Unique suffix: 50 tokens
Cache stats after all requests:
  Hit rate:             90.0%
  Total cached blocks:  26
  Num nodes:            11
```

---

## Running the tests

```bash
uv run pytest tests/test_phase4.py -v
uv run pytest tests/test_phase4.py -k "part1" -v
uv run pytest tests/test_phase4.py -k "part2" -v
uv run pytest tests/test_phase4.py -k "part3" -v
uv run pytest tests/test_phase4.py -k "part4" -v
uv run pytest tests/test_phase4.py -k "part5" -v
uv run pytest tests/test_phase4.py -k "part6" -v
```
