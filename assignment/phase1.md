# Phase 1: Block-based KV Cache Memory Management

> **预计用时**: 10-15 小时 | **难度**: ★★★☆☆  
> **前置知识**: PyTorch tensor 操作、基本数据结构（链表、哈希表）、OS 虚拟内存概念  
> **需要 GPU**: 否（所有测试可在 CPU 上跑，设置 `device="cpu"`）

### 命名对照表

本项目的类名和模块名与 nano-vLLM / vLLM 对齐，方便你日后读源码：

| vkv-engine (本项目) | nano-vLLM | vLLM | 对应 OS 概念 |
|---------------------|-----------|------|-------------|
| `Block` | `Block` | `PhysicalTokenBlock` | Physical Page Frame |
| `BlockAllocator` | inline in `BlockManager` | `BlockAllocator` | Free Frame List |
| `BlockManager` | `BlockManager` | `BlockSpaceManager` | Physical Memory Manager |
| `Sequence` | `Sequence` | `Sequence` | Process (with Page Table) |
| `SequenceStatus` | `SequenceStatus` | `SequenceStatus` | Process State |
| `Scheduler` (Phase 2) | `Scheduler` | `Scheduler` | Process Scheduler |
| `ModelRunner` (Phase 2) | `ModelRunner` | `ModelRunner` | — |
| `SamplingParams` | `SamplingParams` | `SamplingParams` | — |

---

## 目录

- [Part 0: Background — 为什么需要 Block-based Memory](#part-0-background)
- [Part 1: Understanding KV Cache Shapes (热身)](#part-1-kv-cache-shapes)
- [Part 2: Block — KV Cache 的最小存储单元 (核心)](#part-2-block)
- [Part 3: BlockAllocator — 空闲块管理 (核心)](#part-3-block-allocator)
- [Part 4: BlockManager — 预分配的内存池 (核心)](#part-4-block-manager)
- [Part 5: Sequence — 请求级别的 KV 管理 (核心)](#part-5-sequence)
- [Part 6: LRUEvictor — 内存不够时怎么办 (进阶)](#part-6-eviction)
- [Part 7: Swapper — GPU ↔ CPU Swap (进阶)](#part-7-swap)
- [Part 8: Defragmentation (Bonus)](#part-8-defragmentation)

---

## 如何使用本文档

1. **先读背景**：每个 Part 开头有概念讲解，确保理解再动手写代码
2. **填充代码**：在 `vkv/engine/` 下的 `.py` 文件中找到 `raise NotImplementedError("TODO: ...")` 标记
3. **运行测试**：每完成一个 Part，运行对应的测试验证：
   ```bash
   # 运行单个 Part 的测试
   uv run pytest tests/test_phase1.py -k "part1" -v

   # 运行全部 Phase 1 测试
   uv run pytest tests/test_phase1.py -v
   ```
4. **渐进式开发**：Part 1-5 是核心，必须完成；Part 6-8 是进阶/Bonus

---

<a id="part-0-background"></a>
## Part 0: Background — 为什么需要 Block-based Memory

### 问题：朴素的 KV Cache 存储方式

在最简单的 LLM 推理实现中，每个请求的 KV cache 是一个**连续的、不断增长的 tensor**：

```python
# HuggingFace DynamicCache 的简化逻辑
class NaiveKVCache:
    def update(self, layer_idx, new_key, new_value):
        self.key_cache[layer_idx] = torch.cat(
            [self.key_cache[layer_idx], new_key], dim=2
        )
        # 问题: torch.cat 分配新内存 + 复制旧数据 → 碎片！
```

当并发请求数增加到几十上百时，这种方式会导致：

```
GPU Memory (naive approach):
  ├──[Seq A: 2KB]──[碎片]──[Seq B: 8KB]──[碎片]──[Seq C: 1KB]──[碎片]──┤
  总碎片 = 10KB，虽然空闲 > 10KB 但无法分配一个 10KB 的连续块！
```

### 解法：借鉴 OS 的虚拟内存 → PagedAttention

vLLM 的 PagedAttention 把 OS 分页的思想应用到 KV cache：

```
BlockManager 的内存池 (pre-allocated):
  [BLK 0][BLK 1][BLK 2][BLK 3][BLK 4][BLK 5][BLK 6][BLK 7] ...

Sequence A 的 block_table: [0, 2, 3]      ← 不需要连续！
Sequence B 的 block_table: [1, 5, 6]

每个 Block 大小固定（如 16 tokens），按需分配。
请求结束后 Block 归还池子，不会产生碎片。
```

nano-vLLM 用 ~200 行代码实现了这套机制。我们的 Phase 1 把它拆成 5 个步骤让你理解每一层。

---

<a id="part-1-kv-cache-shapes"></a>
## Part 1: Understanding KV Cache Shapes [热身]

> **文件**: `vkv/engine/block.py` 中的 `kv_cache_size_per_token()`  
> **测试**: `uv run pytest tests/test_phase1.py -k "part1" -v`

### Background: KV Cache 的 Tensor 维度

单个 token 在一层 attention 中产出：

```
Key shape:   [num_kv_heads, head_dim]
Value shape: [num_kv_heads, head_dim]
```

总模型：`num_layers × 2(K+V) × num_kv_heads × head_dim × element_size(bytes)`

**GQA 注意**：Llama 3 用 GQA，KV heads (8) 远少于 Q heads (32)。

### 计算示例

```
Llama-3.1-8B: 32 × 2 × 8 × 128 × 2 = 131,072 bytes = 128 KB / token
  1024 tokens = 128 MB,  8192 tokens = 1 GB
```

### Task 1.1: 实现 `kv_cache_size_per_token()`
### Task 1.2: 实现 `num_blocks_for_tokens()`
### Task 1.3: 实现 `kv_block_size_bytes()`

---

<a id="part-2-block"></a>
## Part 2: Block — KV Cache 的最小存储单元 [核心]

> **文件**: `vkv/engine/block.py` 中的 `Block` 类  
> **测试**: `uv run pytest tests/test_phase1.py -k "part2" -v`

### Background

在 nano-vLLM 中，`Block` 是一个很轻量的类：

```python
# nano-vLLM 的 Block（只有元数据）
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1          # 用于 prefix caching
        self.token_ids = []
```

nano-vLLM 的 KV 张量存在 `ModelRunner` 的预分配池中，Block 只是索引。

我们的 Part 2 让 Block 持有自己的 tensor（方案 A），这样你可以直接看到数据读写。
Part 4 会切换到预分配池方案（方案 B），与 nano-vLLM 的实际做法一致。

### Block 的内部结构

```
一个 Block (block_size=16):
┌──────────────────────────────────────────────────────────┐
│ Layer 0: Key   [num_kv_heads, 16, head_dim]              │
│ Layer 0: Value [num_kv_heads, 16, head_dim]              │
│ ...                                                       │
│ Layer N: Key   [num_kv_heads, 16, head_dim]              │
│ Layer N: Value [num_kv_heads, 16, head_dim]              │
└──────────────────────────────────────────────────────────┘
 16 个 slot，slot 0~12 已填充，slot 13~15 等待新 token
```

### Task 2.1: 实现 `Block.__init__()` — 分配 KV tensor 存储
### Task 2.2: 实现 `Block.write_slot()` — 写入单个 token 的 KV
### Task 2.3: 实现 `Block.read_slot()` — 读取 KV
### Task 2.4: 实现 `Block.clear()` — 重置（类比 nano-vLLM 的 `Block.reset()`）

---

<a id="part-3-block-allocator"></a>
## Part 3: BlockAllocator — 空闲块管理 [核心]

> **文件**: `vkv/engine/block_allocator.py`  
> **测试**: `uv run pytest tests/test_phase1.py -k "part3" -v`

### Background

在 nano-vLLM 中，空闲列表管理直接内嵌在 `BlockManager` 里：

```python
# nano-vLLM 的 BlockManager.__init__
self.free_block_ids: deque[int] = deque(range(num_blocks))
self.used_block_ids: set[int] = set()
```

我们把它抽成独立的 `BlockAllocator`（对齐 vLLM 的 `BlockAllocator` 接口），
这样你可以单独测试分配逻辑，然后在 Part 4 组合进 `BlockManager`。

### Free List 工作原理

```
初始:     free = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
allocate(3): → [7, 8, 9],  free = [0, 1, 2, 3, 4, 5, 6]
allocate(2): → [5, 6],     free = [0, 1, 2, 3, 4]
free([8,9]): →             free = [0, 1, 2, 3, 4, 8, 9]
```

### Task 3.1: 实现 `BlockAllocator.__init__()` — 初始化空闲列表
### Task 3.2: 实现 `BlockAllocator.allocate()` — 分配 block IDs
### Task 3.3: 实现 `BlockAllocator.free()` — 归还（检测 double free）
### Task 3.4: 实现 `num_free` 和 `num_used` 属性

---

<a id="part-4-block-manager"></a>
## Part 4: BlockManager — 预分配的内存池 [核心]

> **文件**: `vkv/engine/block_manager.py`  
> **测试**: `uv run pytest tests/test_phase1.py -k "part4" -v`

### Background

这是 nano-vLLM 的 `BlockManager` 对应物。关键区别：

**nano-vLLM**: KV 张量在 `ModelRunner` 中预分配，BlockManager 只管元数据
**vkv-engine**: `BlockManager` 同时管元数据和 KV 张量 — 更接近 vLLM Worker 层的做法

预分配池的布局：

```python
# 启动时一次性分配（nano-vLLM 在 ModelRunner 中做）
for layer in range(num_layers):
    gpu_key_cache[layer] = torch.zeros(
        num_blocks, num_kv_heads, block_size, head_dim,
        dtype=float16, device="cuda"
    )
    gpu_value_cache[layer] = torch.zeros(...)

# Block #5, Layer #3, Slot #7 的 Key:
#   gpu_key_cache[3][5, :, 7, :]  → shape [num_kv_heads, head_dim]
```

### nano-vLLM 中的关键方法对照

| nano-vLLM | vkv-engine | 说明 |
|-----------|------------|------|
| `BlockManager.allocate(seq)` | `BlockManager.allocate(num_blocks)` | 分配物理块 |
| `BlockManager.deallocate(seq)` | `BlockManager.free(block_ids)` | 释放（含引用计数） |
| `BlockManager.can_allocate(seq)` | `BlockManager.can_allocate(num_blocks)` | 检查是否够分配 |
| `BlockManager.may_append(seq)` | `Sequence.append_token()` 内部调用 | decode 时追加 block |
| `BlockManager._allocate_block(id)` | `BlockAllocator.allocate()` | 底层分配 |
| `BlockManager._deallocate_block(id)` | `BlockAllocator.free()` | 底层释放 |

### Task 4.1: 实现 `BlockManager.__init__()` — 预分配 KV 张量池
### Task 4.2: 实现 `BlockManager.allocate()` — 分配 + 引用计数初始化
### Task 4.3: 实现 `BlockManager.free()` — 引用计数减 1，归零时释放
### Task 4.4: 实现 `BlockManager.write_kv()` / `read_kv()`
### Task 4.5: 实现 `BlockManager.gather_kv()` — 拼接多个 block 的 KV
### Task 4.6: 实现 `BlockManager.copy_block()` — COW 复制

---

<a id="part-5-sequence"></a>
## Part 5: Sequence — 请求级别的 KV 管理 [核心]

> **文件**: `vkv/engine/sequence.py`  
> **测试**: `uv run pytest tests/test_phase1.py -k "part5" -v`

### Background

nano-vLLM 的 `Sequence` 是核心数据结构，同时持有：
- **请求元数据**: `seq_id`, `status`, `token_ids`, `sampling_params`
- **块映射表**: `block_table` — 就是 OS 页表

```python
# nano-vLLM 的 Sequence
class Sequence:
    def __init__(self, token_ids, sampling_params):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.block_table = []         # ← 这就是"页表"
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
```

我们的 `Sequence` 在此基础上增加了 `allocate()`, `append_token()`, `fork()`, `free()`
方法，把 nano-vLLM 散布在 `BlockManager` 和 `Scheduler` 中的逻辑聚合到 Sequence 自身。

### SequenceStatus 状态机

```
               ┌──────────────┐
  new request  │   WAITING    │  nano-vLLM: SequenceStatus.WAITING
               └──────┬───────┘
                      │ scheduler picks it, allocate blocks
               ┌──────▼───────┐
               │   RUNNING    │  nano-vLLM: SequenceStatus.RUNNING
               └──┬───────┬───┘
       preempted  │       │  finished
               ┌──▼──┐   │
               │SWAP- │   │  nano-vLLM: no swap, just re-queue as WAITING
               │PED   │   │  vkv: KV saved to CPU memory
               └──┬───┘   │
       resumed    │       │
               ┌──▼───────▼───┐
               │   FINISHED   │  nano-vLLM: SequenceStatus.FINISHED
               └──────────────┘
```

### append_token 对比

```python
# nano-vLLM: 分两步
# Step 1: Sequence.append_token(token_id)
seq.token_ids.append(token_id)
seq.num_tokens += 1

# Step 2: BlockManager.may_append(seq)  — 可能分配新 block
if len(seq) % block_size == 1:   # 新 block 的第一个 token
    block_id = self.free_block_ids[0]
    self._allocate_block(block_id)
    seq.block_table.append(block_id)

# vkv-engine: 合并成一步
block_id, slot_idx = seq.append_token(token_id)
# 内部自动处理新 block 分配，并返回写入位置
```

### Task 5.1: 实现 `Sequence.__init__()` — 初始化请求状态
### Task 5.2: 实现 `Sequence.allocate()` — 为 prefill 分配 blocks
### Task 5.3: 实现 `Sequence.append_token()` — decode 时追加 token
### Task 5.4: 实现 `Sequence.fork()` — COW 前缀共享
### Task 5.5: 实现 `Sequence.free()` — 释放所有 blocks

---

<a id="part-6-eviction"></a>
## Part 6: LRUEvictor — 内存不够时怎么办 [进阶]

> **文件**: `vkv/engine/evictor.py`  
> **测试**: `uv run pytest tests/test_phase1.py -k "part6" -v`

### Background

nano-vLLM 的 preemption 非常简单：

```python
# nano-vLLM Scheduler.preempt
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)     # 直接丢弃 KV cache
    self.waiting.appendleft(seq)           # 重新排队，之后重新 prefill
```

这意味着被抢占的序列需要**重新从头计算 prefill**，浪费了之前的计算。

我们的 LRUEvictor 更智能：追踪访问顺序，选择最"冷"的序列淘汰。
配合 Part 7 的 Swapper，被淘汰的 KV cache 可以保存到 CPU 而非直接丢弃。

### Task 6.1: 实现 `LRUEvictor.__init__()` / `add()` / `remove()` / `touch()`
### Task 6.2: 实现 `LRUEvictor.evict()` — 淘汰最旧的序列

---

<a id="part-7-swap"></a>
## Part 7: Swapper — GPU ↔ CPU Swap [进阶]

> **文件**: `vkv/engine/swapper.py`  
> **测试**: `uv run pytest tests/test_phase1.py -k "part7" -v`

### Background

nano-vLLM 没有 swap。被 preempt 的序列 KV cache 直接丢弃。

vLLM 有完整的 swap 机制，通过 PCIe 将 KV cache 在 GPU ↔ CPU 之间搬运。
这是 vkv-engine 比 nano-vLLM 更深入的地方之一。

### Task 7.1: 实现 `Swapper.swap_out()` — GPU → CPU
### Task 7.2: 实现 `Swapper.swap_in()` — CPU → GPU

---

<a id="part-8-defragmentation"></a>
## Part 8: Defragmentation [Bonus]

> **文件**: `vkv/engine/block_manager.py` 的 `compute_fragmentation()`  
> **测试**: `uv run pytest tests/test_phase1.py -k "part8" -v`

### Task 8.1: 实现 `BlockManager.compute_fragmentation()` — 计算内部碎片率

---

## 运行测试

```bash
# 运行全部
uv run pytest tests/test_phase1.py -v

# 按 Part 运行
uv run pytest tests/test_phase1.py -k "part1" -v
uv run pytest tests/test_phase1.py -k "part2" -v
uv run pytest tests/test_phase1.py -k "part3" -v
uv run pytest tests/test_phase1.py -k "part4" -v
uv run pytest tests/test_phase1.py -k "part5" -v
uv run pytest tests/test_phase1.py -k "part6" -v
uv run pytest tests/test_phase1.py -k "part7" -v
uv run pytest tests/test_phase1.py -k "part8" -v

# 单个测试
uv run pytest tests/test_phase1.py::TestPart4::test_pool_write_read_kv -v
```
