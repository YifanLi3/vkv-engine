# Phase 2: Continuous Batching Scheduler

> **Est. time**: 10–15 hours | **Difficulty**: ★★★★☆
> **Prerequisites**: Phase 1 fully done; familiarity with queues / state machines
> **Requires GPU**: No (all tests use `MockModelRunner`)

### Naming reference

| vkv-engine | nano-vLLM | vLLM |
|------------|-----------|------|
| `Scheduler` | `Scheduler` | `Scheduler` |
| `Scheduler.schedule()` | `Scheduler.schedule()` | `Scheduler.schedule()` |
| `Scheduler.add()` | `Scheduler.add()` | `Scheduler.add_seq_group()` |
| `Scheduler.preempt()` | `Scheduler.preempt()` | `Scheduler._preempt()` |
| `Scheduler.postprocess()` | `Scheduler.postprocess()` | inside `LLMEngine.step()` |
| `LLMEngine` | `LLMEngine` | `LLMEngine` |
| `SchedulerOutput` | `(list, bool)` tuple | `SchedulerOutputs` |

---

## Table of Contents

- [Part 0: Background — what is continuous batching](#part-0-background)
- [Part 1: SchedulerConfig & SchedulerOutput (warm-up)](#part-1-config)
- [Part 2: Basic scheduling — prefill first (core)](#part-2-basic-schedule)
- [Part 3: Decode scheduling + preemption (core)](#part-3-decode-preemption)
- [Part 4: Postprocess — handle generation results (core)](#part-4-postprocess)
- [Part 5: LLMEngine — wire everything together (core)](#part-5-llm-engine)
- [Part 6: Chunked prefill (advanced)](#part-6-chunked-prefill)
- [Part 7: End-to-end simulated inference (integration)](#part-7-e2e)

---

## How to use this document

1. **Read the background first**: understand continuous batching vs static batching.
2. **Fill in the code**: implement the TODOs in `vkv/engine/scheduler.py` and `vkv/engine/llm_engine.py`.
3. **Run the tests**:
   ```bash
   uv run pytest tests/test_phase2.py -k "part1" -v
   uv run pytest tests/test_phase2.py -v     # everything
   ```

---

<a id="part-0-background"></a>
## Part 0: Background — What is continuous batching

### Static batching (the traditional way)

```
Batch = [Seq A (100 tokens), Seq B (50 tokens), Seq C (200 tokens)]

Every step: all sequences decode together
Step 1:   A generates a token, B generates a token, C generates a token
Step 2:   A generates a token, B generates a token, C generates a token
...
Step 50:  A generates a token, B is done!         C generates a token
Step 51:  A generates a token, B [idle wait]      C generates a token   ← B's GPU compute wasted
...
Step 100: A is done!            B [idle wait]      C generates a token   ← A and B both idle
...
Step 200: A [idle wait]         B [idle wait]      C is done!            ← finally

Problem: after short requests finish, the GPU spins idle waiting on the longest one.
```

### Continuous batching (nano-vLLM / vLLM way)

```
Step 1:   A generates a token, B generates a token, C generates a token
...
Step 50:  A generates a token, B is done! → immediately evicted, D joins!   C generates a token
Step 51:  A generates a token, D runs prefill                               C generates a token
                                 ↑ new request fills the slot immediately
...
Step 100: A is done! → E joins   D generates a token                        C generates a token

GPU stays saturated with no idle time.
```

### Core scheduling questions

Every `schedule()` call must decide:

1. **Prefill or decode?** — prefill is compute-bound, decode is memory-bound
2. **Which seqs enter the batch?** — bounded by GPU memory and `max_num_seqs`
3. **What if memory runs out?** — preempt some running seqs

### nano-vLLM's Scheduler core logic (~60 lines)

```python
# Simplified nano-vLLM scheduler.py
def schedule(self):
    # prefill first
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

    # no prefill left, do decode
    for seq in self.running:
        if not can_append(seq):
            preempt(self.running.pop())  # OOM, drop the last one
        else:
            scheduled.append(seq)
    return scheduled, False  # is_prefill = False
```

Our Phase 2 adds on top of this:
- **SWAPPED state + swap in/out** (don't discard KV)
- **LRUEvictor** (smarter eviction choice)
- **max_num_batched_tokens** (control batch size)
- **Chunked prefill** (Part 6 advanced)

---

<a id="part-1-config"></a>
## Part 1: SchedulerConfig & SchedulerOutput [warm-up]

> **File**: `vkv/engine/scheduler.py`
> **Tests**: `uv run pytest tests/test_phase2.py -k "part1" -v`

### Task 1.1: Implement `SchedulerConfig`

```python
@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256            # max sequences in a batch
    max_num_batched_tokens: int = 4096 # max tokens processed in a batch
```

### Task 1.2: Implement `SchedulerOutput`

nano-vLLM just returns a `(list, bool)` tuple. A dataclass is clearer:

```python
@dataclass
class SchedulerOutput:
    scheduled_seqs: List[Sequence]  # sequences to run this step
    is_prefill: bool                # True = prefill, False = decode
    preempted_seqs: List[Sequence]  # sequences preempted this step
    swapped_in_seqs: List[Sequence] # sequences swapped back in this step
```

---

<a id="part-2-basic-schedule"></a>
## Part 2: Basic scheduling — prefill first [core]

> **File**: `_schedule_prefill()` in `vkv/engine/scheduler.py`
> **Tests**: `uv run pytest tests/test_phase2.py -k "part2" -v`

### Background

Scheduling's top priority is **prefill**: move requests from the WAITING queue to the GPU.

```
When schedule() is called:

  waiting: [Seq D, Seq E, Seq F]     ← waiting for prefill
  running: [Seq A, Seq B, Seq C]     ← currently decoding

  Check waiting queue first, prefill what we can.
  If nothing to prefill, do decode.
```

### Task 2.1: Implement `_schedule_prefill()`

```
Algorithm:
1. Iterate the waiting queue
2. For each seq, check:
   a. num_seqs_in_batch < max_num_seqs
   b. total_tokens_in_batch + seq_len <= max_num_batched_tokens
   c. BlockManager has enough free blocks (can_allocate)
3. If OK, allocate blocks and move to the running queue
4. Return the list of sequences to prefill this round
```

### Task 2.2: Implement the prefill part of `schedule()`

```python
def schedule(self) -> SchedulerOutput:
    # 1. Try prefill first
    prefill_seqs = self._schedule_prefill()
    if prefill_seqs:
        return SchedulerOutput(
            scheduled_seqs=prefill_seqs,
            is_prefill=True,
            preempted_seqs=[],
            swapped_in_seqs=[],
        )

    # 2. No prefill, do decode (Part 3)
    ...
```

---

<a id="part-3-decode-preemption"></a>
## Part 3: Decode scheduling + preemption [core]

> **File**: `_schedule_decode()` and `preempt()` in `vkv/engine/scheduler.py`
> **Tests**: `uv run pytest tests/test_phase2.py -k "part3" -v`

### Background: why decode needs preemption

In decode, each seq produces one token per step and may need a new block:

```
Seq A: uses [Block 3, Block 7]; Block 7 just filled up.
       Next token needs a new block → but no free block left on GPU!

Solution: preempt another sequence, release its blocks.
```

### Preemption: nano-vLLM vs vkv

```python
# nano-vLLM: drop KV, re-queue
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)

# vkv-engine: swap to CPU, keep KV
def preempt(self, seq):
    mapping = self.swapper.swap_out(seq.block_table)
    seq.cpu_block_table = mapping       # remember CPU-side block IDs
    seq.status = SequenceStatus.SWAPPED
    self.swapped.append(seq)
```

### Task 3.1: Implement `preempt()`

Two modes:
- `mode="recompute"`: nano-vLLM style — drop KV, re-queue
- `mode="swap"`: vkv extension — swap to CPU

### Task 3.2: Implement `_schedule_decode()`

```
Algorithm:
1. Iterate the running queue
2. For each seq, check: does appending the next token require a new block?
3. If a new block is needed and GPU has none → preempt another seq
4. Allocate the block (through Sequence.append_token internally)
5. Return the list of seqs to decode this round
```

### Task 3.3: Implement `_try_swap_in()`

Before decoding, try to bring SWAPPED seqs back to GPU:

```
Algorithm:
1. Check the swapped queue
2. If there are enough GPU blocks, swap_in and move back to the running queue
3. Return the list of swapped-in seqs
```

---

<a id="part-4-postprocess"></a>
## Part 4: Postprocess — handle generation results [core]

> **File**: `postprocess()` in `vkv/engine/scheduler.py`
> **Tests**: `uv run pytest tests/test_phase2.py -k "part4" -v`

### Background

nano-vLLM's `postprocess` runs after every decode step:

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

### Task 4.1: Implement `postprocess()`

```
For each seq and its generated token:
1. Call seq.append_token(token_id)
2. Check whether it's finished (EOS or reached max_tokens)
3. If finished → seq.free(), remove from running
```

---

<a id="part-5-llm-engine"></a>
## Part 5: LLMEngine — wire everything together [core]

> **File**: `vkv/engine/llm_engine.py` (new file)
> **Tests**: `uv run pytest tests/test_phase2.py -k "part5" -v`

### Background

nano-vLLM's `LLMEngine` is the top-level coordinator:

```python
# nano-vLLM LLMEngine.step()
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids)
```

One `step()` = one round of schedule + one model call + result handling.

### Task 5.1: Implement `LLMEngine.__init__()`

```python
class LLMEngine:
    def __init__(self, model_config, cache_config, scheduler_config):
        self.block_manager = BlockManager(model_config, cache_config)
        self.scheduler = Scheduler(self.block_manager, scheduler_config)
        self.model_runner = MockModelRunner(model_config)
```

### Task 5.2: Implement `LLMEngine.add_request()`

```python
def add_request(self, token_ids, sampling_params=None):
    seq = Sequence(token_ids, self.block_manager, sampling_params)
    self.scheduler.add(seq)
    return seq.seq_id
```

### Task 5.3: Implement `LLMEngine.step()`

One step of the core loop: schedule → execute → postprocess.

### Task 5.4: Implement `LLMEngine.generate()`

Run `step()` until every request finishes, then collect outputs.

---

<a id="part-6-chunked-prefill"></a>
## Part 6: Chunked prefill [advanced]

> **File**: `vkv/engine/scheduler.py`
> **Tests**: `uv run pytest tests/test_phase2.py -k "part6" -v`

### Background: why chunked prefill

A long prompt's prefill blocks all decode requests:

```
Regular prefill:
  Step N:   Seq A (2048-token prompt) runs prefill ← very long
  Step N+1: Seq B, C, D decode is blocked          ← TPOT spikes

Chunked prefill (Sarathi-style):
  Step N:    First 512 tokens of Seq A prefill + Seq B, C, D decode
  Step N+1:  Next 512 tokens of Seq A prefill  + Seq B, C, D decode
  Step N+2:  Next 512 tokens of Seq A prefill  + Seq B, C, D decode
  Step N+3:  Last 512 tokens of Seq A prefill  + Seq B, C, D decode
  → Decode latency is unaffected
```

### Task 6.1: Implement `_schedule_chunked_prefill()`

Mix a prefill chunk with decodes in the same batch:

```
budget = max_num_batched_tokens
1. Schedule decodes first (each seq takes 1 token) → budget -= num_decode_seqs
2. Use remaining budget for a prefill chunk
```

---

<a id="part-7-e2e"></a>
## Part 7: End-to-end simulated inference [integration]

> **File**: integration tests in `tests/test_phase2.py`
> **Tests**: `uv run pytest tests/test_phase2.py -k "part7" -v`

Use `LLMEngine` + `MockModelRunner` to simulate a full inference flow:
- Add multiple requests
- Run the `step()` loop
- Assert that every request eventually completes
- Assert that the BlockManager has no memory leak

---

## Running the tests

```bash
uv run pytest tests/test_phase2.py -v                # everything
uv run pytest tests/test_phase2.py -k "part1" -v     # single Part
uv run pytest tests/test_phase2.py -k "part2" -v
uv run pytest tests/test_phase2.py -k "part3" -v
uv run pytest tests/test_phase2.py -k "part4" -v
uv run pytest tests/test_phase2.py -k "part5" -v
uv run pytest tests/test_phase2.py -k "part6" -v
uv run pytest tests/test_phase2.py -k "part7" -v
```
