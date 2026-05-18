# Phase 2: Continuous Batching Scheduler

> **预计用时**: 10-15 小时 | **难度**: ★★★★☆  
> **前置知识**: Phase 1 全部完成、队列/状态机概念  
> **需要 GPU**: 否（所有测试使用 MockModelRunner）

### 命名对照表

| vkv-engine | nano-vLLM | vLLM |
|------------|-----------|------|
| `Scheduler` | `Scheduler` | `Scheduler` |
| `Scheduler.schedule()` | `Scheduler.schedule()` | `Scheduler.schedule()` |
| `Scheduler.add()` | `Scheduler.add()` | `Scheduler.add_seq_group()` |
| `Scheduler.preempt()` | `Scheduler.preempt()` | `Scheduler._preempt()` |
| `Scheduler.postprocess()` | `Scheduler.postprocess()` | 在 `LLMEngine.step()` 中 |
| `LLMEngine` | `LLMEngine` | `LLMEngine` |
| `SchedulerOutput` | `(list, bool)` tuple | `SchedulerOutputs` |

---

## 目录

- [Part 0: Background — 什么是 Continuous Batching](#part-0-background)
- [Part 1: SchedulerConfig & SchedulerOutput (热身)](#part-1-config)
- [Part 2: 基本调度 — Prefill 优先 (核心)](#part-2-basic-schedule)
- [Part 3: Decode 调度 + Preemption (核心)](#part-3-decode-preemption)
- [Part 4: Postprocess — 处理生成结果 (核心)](#part-4-postprocess)
- [Part 5: LLMEngine — 串联所有组件 (核心)](#part-5-llm-engine)
- [Part 6: Chunked Prefill (进阶)](#part-6-chunked-prefill)
- [Part 7: 端到端模拟推理 (集成)](#part-7-e2e)

---

## 如何使用本文档

1. **先读背景**：理解 continuous batching 和 static batching 的区别
2. **填充代码**：在 `vkv/engine/scheduler.py` 和 `vkv/engine/llm_engine.py` 中实现 TODO
3. **运行测试**：
   ```bash
   uv run pytest tests/test_phase2.py -k "part1" -v
   uv run pytest tests/test_phase2.py -v     # 全部
   ```

---

<a id="part-0-background"></a>
## Part 0: Background — 什么是 Continuous Batching

### Static Batching（传统方式）

```
Batch = [Seq A (100 tokens), Seq B (50 tokens), Seq C (200 tokens)]

每一步：所有 Sequence 一起 decode
Step 1: A 生成 token, B 生成 token, C 生成 token
Step 2: A 生成 token, B 生成 token, C 生成 token
...
Step 50: A 生成 token, B 完成了！  C 生成 token
Step 51: A 生成 token, B [空等]    C 生成 token   ← B 的 GPU 算力浪费了
...
Step 100: A 完成了！   B [空等]    C 生成 token   ← A 和 B 都在浪费
...
Step 200: A [空等]     B [空等]    C 完成了！      ← 终于全部完成

问题：短请求完成后 GPU 在空转，等最长的请求
```

### Continuous Batching（nano-vLLM / vLLM 的方式）

```
Step 1: A 生成 token, B 生成 token, C 生成 token
...
Step 50: A 生成 token, B 完成！→ 立刻移出, D 加入！  C 生成 token
Step 51: A 生成 token, D 做 prefill                   C 生成 token
                       ↑ 新请求立刻填补空位
...
Step 100: A 完成！→ E 加入  D 生成 token  C 生成 token

GPU 永远满载，没有空转
```

### 调度的核心问题

每个 `schedule()` 调用需要决定：

1. **做 prefill 还是 decode？** — prefill 是 compute-bound，decode 是 memory-bound
2. **选哪些序列进入 batch？** — 受限于 GPU 显存和 max_num_seqs
3. **显存不够怎么办？** — preempt（淘汰）某些 running 序列

### nano-vLLM 的 Scheduler 核心逻辑（~60 行）

```python
# nano-vLLM scheduler.py 简化版
def schedule(self):
    # 优先 prefill
    while self.waiting:
        seq = self.waiting[0]
        if can_allocate(seq):
            allocate(seq)
            seq.status = RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled.append(seq)
    if scheduled:
        return scheduled, True   # is_prefill = True

    # 没有 prefill，做 decode
    for seq in self.running:
        if not can_append(seq):
            preempt(self.running.pop())  # 内存不够，淘汰最后一个
        else:
            scheduled.append(seq)
    return scheduled, False  # is_prefill = False
```

我们的 Phase 2 在此基础上增加：
- **SWAPPED 状态 + swap in/out**（不丢弃 KV）
- **LRUEvictor**（智能选择淘汰对象）
- **max_num_batched_tokens**（控制 batch 大小）
- **Chunked Prefill**（Part 6 进阶）

---

<a id="part-1-config"></a>
## Part 1: SchedulerConfig & SchedulerOutput [热身]

> **文件**: `vkv/engine/scheduler.py`  
> **测试**: `uv run pytest tests/test_phase2.py -k "part1" -v`

### Task 1.1: 实现 `SchedulerConfig`

```python
@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256            # batch 中最多几个 sequence
    max_num_batched_tokens: int = 4096 # 一个 batch 最多处理多少 token
```

### Task 1.2: 实现 `SchedulerOutput`

nano-vLLM 直接返回 `(list, bool)` tuple。我们用一个 dataclass 更清晰：

```python
@dataclass
class SchedulerOutput:
    scheduled_seqs: List[Sequence]  # 本轮要执行的序列
    is_prefill: bool                # True = prefill, False = decode
    preempted_seqs: List[Sequence]  # 被抢占的序列
    swapped_in_seqs: List[Sequence] # 被 swap in 的序列
```

---

<a id="part-2-basic-schedule"></a>
## Part 2: 基本调度 — Prefill 优先 [核心]

> **文件**: `vkv/engine/scheduler.py` 的 `_schedule_prefill()`  
> **测试**: `uv run pytest tests/test_phase2.py -k "part2" -v`

### Background

调度的第一优先级是 **prefill**：把 WAITING 队列中的请求搬到 GPU 上。

```
schedule() 被调用时：

  waiting: [Seq D, Seq E, Seq F]     ← 等待 prefill
  running: [Seq A, Seq B, Seq C]     ← 正在 decode

  先检查 waiting 队列，能 prefill 就 prefill
  没有可 prefill 的，再做 decode
```

### Task 2.1: 实现 `_schedule_prefill()`

```
算法：
1. 遍历 waiting 队列
2. 对每个 seq，检查：
   a. batch 中序列数 < max_num_seqs
   b. batch 总 token 数 + seq 长度 <= max_num_batched_tokens
   c. BlockManager 有足够的空闲 block（can_allocate）
3. 如果满足，分配 block，移到 running 队列
4. 返回本轮要 prefill 的序列列表
```

### Task 2.2: 实现 `schedule()` 的 prefill 部分

```python
def schedule(self) -> SchedulerOutput:
    # 1. 先尝试 prefill
    prefill_seqs = self._schedule_prefill()
    if prefill_seqs:
        return SchedulerOutput(
            scheduled_seqs=prefill_seqs,
            is_prefill=True,
            preempted_seqs=[],
            swapped_in_seqs=[],
        )

    # 2. 没有 prefill，做 decode（Part 3）
    ...
```

---

<a id="part-3-decode-preemption"></a>
## Part 3: Decode 调度 + Preemption [核心]

> **文件**: `vkv/engine/scheduler.py` 的 `_schedule_decode()` 和 `preempt()`  
> **测试**: `uv run pytest tests/test_phase2.py -k "part3" -v`

### Background: 为什么 Decode 需要 Preemption

Decode 阶段每个序列每步生成一个 token，可能需要新 block：

```
Seq A: 已经用了 [Block 3, Block 7]，Block 7 刚好满了
       下一个 token 需要新 block → 但 GPU 没有空闲 block 了！

解决：preempt（抢占）另一个序列，释放它的 block
```

### nano-vLLM vs vkv 的 Preemption 对比

```python
# nano-vLLM: 直接丢弃 KV，重新排队
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)

# vkv-engine: swap to CPU，保留 KV
def preempt(self, seq):
    mapping = self.swapper.swap_out(seq.block_table)
    seq.cpu_block_table = mapping       # 记住 CPU 端的 block 位置
    seq.status = SequenceStatus.SWAPPED
    self.swapped.append(seq)
```

### Task 3.1: 实现 `preempt()`

两种模式：
- `mode="recompute"`: nano-vLLM 风格，丢弃 KV，重新排队
- `mode="swap"`: vkv 扩展，swap to CPU

### Task 3.2: 实现 `_schedule_decode()`

```
算法：
1. 遍历 running 队列中的序列
2. 对每个 seq，检查：append 下一个 token 是否需要新 block
3. 如果需要新 block 且 GPU 没有空闲 block → preempt 其他序列
4. 分配 block（通过 Sequence.append_token 内部处理）
5. 返回本轮要 decode 的序列列表
```

### Task 3.3: 实现 `_try_swap_in()`

在 decode 之前，尝试把 SWAPPED 的序列搬回 GPU：

```
算法：
1. 检查 swapped 队列
2. 如果有足够的 GPU block，swap_in 并移回 running 队列
3. 返回 swap in 的序列列表
```

---

<a id="part-4-postprocess"></a>
## Part 4: Postprocess — 处理生成结果 [核心]

> **文件**: `vkv/engine/scheduler.py` 的 `postprocess()`  
> **测试**: `uv run pytest tests/test_phase2.py -k "part4" -v`

### Background

nano-vLLM 的 `postprocess` 在每步 decode 后调用：

```python
# nano-vLLM
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

### Task 4.1: 实现 `postprocess()`

```
对每个序列和它生成的 token：
1. 调用 seq.append_token(token_id)
2. 检查是否结束（EOS 或达到 max_tokens）
3. 如果结束 → seq.free(), 从 running 移除
```

---

<a id="part-5-llm-engine"></a>
## Part 5: LLMEngine — 串联所有组件 [核心]

> **文件**: `vkv/engine/llm_engine.py`（新文件）  
> **测试**: `uv run pytest tests/test_phase2.py -k "part5" -v`

### Background

nano-vLLM 的 `LLMEngine` 是顶层协调者：

```python
# nano-vLLM LLMEngine.step()
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids)
```

一个 `step()` = 一轮调度 + 一次模型计算 + 结果处理。

### Task 5.1: 实现 `LLMEngine.__init__()`

```python
class LLMEngine:
    def __init__(self, model_config, cache_config, scheduler_config):
        self.block_manager = BlockManager(model_config, cache_config)
        self.scheduler = Scheduler(self.block_manager, scheduler_config)
        self.model_runner = MockModelRunner(model_config)
```

### Task 5.2: 实现 `LLMEngine.add_request()`

```python
def add_request(self, token_ids, sampling_params=None):
    seq = Sequence(token_ids, self.block_manager, sampling_params)
    self.scheduler.add(seq)
    return seq.seq_id
```

### Task 5.3: 实现 `LLMEngine.step()`

核心循环的一步。调度 → 执行 → 后处理。

### Task 5.4: 实现 `LLMEngine.generate()`

运行 `step()` 直到所有请求完成，收集输出。

---

<a id="part-6-chunked-prefill"></a>
## Part 6: Chunked Prefill [进阶]

> **文件**: `vkv/engine/scheduler.py`  
> **测试**: `uv run pytest tests/test_phase2.py -k "part6" -v`

### Background: 为什么需要 Chunked Prefill

长 prompt 的 prefill 会阻塞所有 decode 请求：

```
普通 prefill:
  Step N:   Seq A (2048 token prompt) 做 prefill ← 耗时很长
  Step N+1: Seq B, C, D 的 decode 被阻塞         ← TPOT 飙升

Chunked prefill (Sarathi 思路):
  Step N:   Seq A 的前 512 token prefill + Seq B, C, D decode
  Step N+1: Seq A 的下 512 token prefill + Seq B, C, D decode
  Step N+2: Seq A 的下 512 token prefill + Seq B, C, D decode
  Step N+3: Seq A 的最后 512 token prefill + Seq B, C, D decode
  → Decode 延迟不受影响
```

### Task 6.1: 实现 `_schedule_chunked_prefill()`

在同一个 batch 中混合 prefill chunk 和 decode：

```
budget = max_num_batched_tokens
1. 先安排 decode（每个 seq 占 1 token）→ budget -= num_decode_seqs
2. 用剩余 budget 做 prefill chunk
```

---

<a id="part-7-e2e"></a>
## Part 7: 端到端模拟推理 [集成]

> **文件**: `tests/test_phase2.py` 的集成测试  
> **测试**: `uv run pytest tests/test_phase2.py -k "part7" -v`

使用 `LLMEngine` + `MockModelRunner` 模拟完整的推理流程：
- 添加多个请求
- 运行 step() 循环
- 验证所有请求最终完成
- 验证 BlockManager 没有内存泄漏

---

## 运行测试

```bash
uv run pytest tests/test_phase2.py -v                # 全部
uv run pytest tests/test_phase2.py -k "part1" -v     # 单个 Part
uv run pytest tests/test_phase2.py -k "part2" -v
uv run pytest tests/test_phase2.py -k "part3" -v
uv run pytest tests/test_phase2.py -k "part4" -v
uv run pytest tests/test_phase2.py -k "part5" -v
uv run pytest tests/test_phase2.py -k "part6" -v
uv run pytest tests/test_phase2.py -k "part7" -v
```
