# Phase 3: KV Cache Quantization

> **Est. time**: 8–12 hours | **Difficulty**: ★★★★☆
> **Prerequisites**: Phase 1 done; basic knowledge of floating-point / integer representation
> **Requires GPU**: No (all tests run on CPU)

---

## Table of Contents

- [Part 0: Background — Why quantize the KV cache](#part-0-background)
- [Part 1: Quantization math basics (warm-up)](#part-1-quantization-math)
- [Part 2: Per-tensor INT8 quantizer (core)](#part-2-per-tensor)
- [Part 3: Per-channel INT8 quantizer (core)](#part-3-per-channel)
- [Part 4: Grouped INT4 quantizer (core)](#part-4-grouped-int4)
- [Part 5: Quantized KV cache integration — QuantizedCacheManager (core)](#part-5-integration)
- [Part 6: Quantization accuracy evaluation (advanced)](#part-6-evaluation)

---

<a id="part-0-background"></a>
## Part 0: Background — Why quantize the KV cache

### Problem: KV cache eats too much GPU memory

```
Llama-3.1-8B, 64 concurrent requests, 8K context each:
  KV cache = 64 × 8192 × 128 KB/token = 64 GB
  Nearly fills an A100 (80 GB)

If we can compress the KV cache to half or a quarter:
  INT8: 64 GB → 32 GB    ← can serve 128 concurrent
  INT4: 64 GB → 16 GB    ← can serve 256 concurrent
```

### Basic idea of quantization

Compress FP16 (16 bit) KV data down to INT8 (8 bit) or INT4 (4 bit):

```
FP16:  2 bytes per value, range ~[-65504, 65504], high precision
INT8:  1 byte per value,  range [-128, 127],       lower precision
INT4:  0.5 byte per value, range [-8, 7],           much lower precision

FP16 values:  [0.123, -0.456, 0.789, -0.012, ...]
                    ↓ quantize
INT8 values:  [16, -58, 101, -2, ...]
                    ↓ dequantize (approximate reconstruction)
FP16 approx:  [0.125, -0.453, 0.789, -0.016, ...]  ← small error, close enough
```

### Quantization formula

```
quantize:    q = round(x / scale)      FP16 → INT8
dequantize:  x̂ = q × scale              INT8 → FP16 (approx)

scale = max(|x|) / 127                 makes INT8 range cover data range
```

### Different quantization granularities

| Granularity | How scale is computed | Precision | Storage overhead |
|-------------|------------------------|-----------|------------------|
| Per-tensor  | One scale for the whole tensor | Lowest | 1 float |
| Per-channel | One scale per head | Medium | `num_heads` floats |
| Per-token   | One scale per token | Higher | `seq_len` floats |
| Grouped     | One scale per `group_size` elements | Highest | Multiple floats |

---

<a id="part-1-quantization-math"></a>
## Part 1: Quantization math basics [warm-up]

> **File**: `vkv/engine/quantizer.py`
> **Tests**: `uv run pytest tests/test_phase3.py -k "part1" -v`

### Task 1.1: Implement `compute_scale()`

Given an FP16 tensor, compute the quantization scale:

```python
scale = max(|x|) / (2^(bits-1) - 1)

# INT8: scale = max(|x|) / 127
# INT4: scale = max(|x|) / 7
```

### Task 1.2: Implement `quantize_tensor()`

Quantize an FP16 tensor to INT8 using the scale:

```python
q = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
```

### Task 1.3: Implement `dequantize_tensor()`

Dequantize back to FP16:

```python
x_approx = q.float() * scale
```

---

<a id="part-2-per-tensor"></a>
## Part 2: Per-tensor INT8 quantizer [core]

> **File**: `PerTensorQuantizer` in `vkv/engine/quantizer.py`
> **Tests**: `uv run pytest tests/test_phase3.py -k "part2" -v`

### Background

The simplest quantization scheme: the whole tensor shares one scale.

```
Input: key tensor [num_kv_heads, seq_len, head_dim] = [8, 100, 128]
       102 400 values in total → find max |value| → one scale

Pros: simple, minimal storage overhead (just 1 extra float)
Cons: if one head has very large values, precision of other heads is wasted
```

### Task 2.1: Implement `PerTensorQuantizer.quantize()`
### Task 2.2: Implement `PerTensorQuantizer.dequantize()`

---

<a id="part-3-per-channel"></a>
## Part 3: Per-channel INT8 quantizer [core]

> **File**: `PerChannelQuantizer` in `vkv/engine/quantizer.py`
> **Tests**: `uv run pytest tests/test_phase3.py -k "part3" -v`

### Background

Each KV head has its own scale, higher precision:

```
Input: [num_kv_heads, seq_len, head_dim] = [8, 100, 128]

Per-tensor:  1 scale  → 102 400 values share it
Per-channel: 8 scales → each head quantized independently

head 0 range: [-0.5, 0.5] → scale_0 = 0.5 / 127 = 0.00394
head 1 range: [-2.0, 2.0] → scale_1 = 2.0 / 127 = 0.01575
                              ↑ each head uses its own scale, no interference
```

### Task 3.1: Implement `PerChannelQuantizer.quantize()`

`scales` shape: `[num_kv_heads]` — one scale per head.

### Task 3.2: Implement `PerChannelQuantizer.dequantize()`

---

<a id="part-4-grouped-int4"></a>
## Part 4: Grouped INT4 quantizer [core]

> **File**: `GroupedQuantizer` in `vkv/engine/quantizer.py`
> **Tests**: `uv run pytest tests/test_phase3.py -k "part4" -v`

### Background

INT4 has only 16 values (-8 to 7), so precision is very low. Compensate by grouping:

```
head_dim = 128, group_size = 32

Every 32 elements form a group with its own scale:
  group 0: elements 0-31    → scale_0
  group 1: elements 32-63   → scale_1
  group 2: elements 64-95   → scale_2
  group 3: elements 96-127  → scale_3

4 groups × 1 float scale each = 4 floats of overhead
Buys 4× compression (FP16 → INT4)
```

### Task 4.1: Implement `GroupedQuantizer.quantize()`
### Task 4.2: Implement `GroupedQuantizer.dequantize()`

Special handling for INT4: PyTorch has no `int4` dtype, so store in `int8` and clamp to [-8, 7].

---

<a id="part-5-integration"></a>
## Part 5: Quantized KV cache integration [core]

> **File**: `QuantizedCacheManager` in `vkv/engine/quantizer.py`
> **Tests**: `uv run pytest tests/test_phase3.py -k "part5" -v`

### Background

Hook the quantizer into BlockManager's write/read path:

```
Normal:
  write_kv(key_fp16) → gpu_key_cache holds FP16

Quantized:
  write_kv(key_fp16) → quantize to INT8 → gpu_key_cache holds INT8 (half the memory)
  read_kv() → read INT8 → dequantize to FP16 → return
```

### Task 5.1: Implement `QuantizedCacheManager.write_quantized()`

Quantize on write:

```python
def write_quantized(self, block_id, layer_idx, slot_idx, key, value):
    q_key, k_scale = self.quantizer.quantize(key)
    q_value, v_scale = self.quantizer.quantize(value)
    # Store the quantized data and scales
```

### Task 5.2: Implement `QuantizedCacheManager.read_dequantized()`

Dequantize on read:

```python
def read_dequantized(self, block_id, layer_idx, slot_idx):
    q_key, k_scale = # ...read quantized data and scale
    key = self.quantizer.dequantize(q_key, k_scale)
    # Return approximate FP16 data
```

---

<a id="part-6-evaluation"></a>
## Part 6: Quantization accuracy evaluation [advanced]

> **File**: `vkv/engine/quantizer.py`
> **Tests**: `uv run pytest tests/test_phase3.py -k "part6" -v`

### Task 6.1: Implement `compute_quantization_error()`

Measure the error introduced by quantization:

```python
def compute_quantization_error(original, reconstructed):
    mse = ((original - reconstructed) ** 2).mean()
    cosine_sim = F.cosine_similarity(original.flatten(), reconstructed.flatten(), dim=0)
    return mse, cosine_sim
```

### Task 6.2: Compare different quantization strategies

Generate random KV data and compare per-tensor vs per-channel vs grouped accuracy.

---

## Running the tests

```bash
uv run pytest tests/test_phase3.py -v
uv run pytest tests/test_phase3.py -k "part1" -v
uv run pytest tests/test_phase3.py -k "part2" -v
uv run pytest tests/test_phase3.py -k "part3" -v
uv run pytest tests/test_phase3.py -k "part4" -v
uv run pytest tests/test_phase3.py -k "part5" -v
uv run pytest tests/test_phase3.py -k "part6" -v
```
