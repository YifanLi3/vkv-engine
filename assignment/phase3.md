# Phase 3: KV Cache Quantization

> **预计用时**: 8-12 小时 | **难度**: ★★★★☆  
> **前置知识**: Phase 1 完成、基本的浮点数/整数表示概念  
> **需要 GPU**: 否（所有测试在 CPU 上跑）

---

## 目录

- [Part 0: Background — 为什么量化 KV Cache](#part-0-background)
- [Part 1: 基础量化数学 (热身)](#part-1-quantization-math)
- [Part 2: Per-tensor INT8 Quantizer (核心)](#part-2-per-tensor)
- [Part 3: Per-channel INT8 Quantizer (核心)](#part-3-per-channel)
- [Part 4: Grouped INT4 Quantizer (核心)](#part-4-grouped-int4)
- [Part 5: 量化 KV Cache 集成 — QuantizedBlockManager (核心)](#part-5-integration)
- [Part 6: 量化精度评估 (进阶)](#part-6-evaluation)

---

<a id="part-0-background"></a>
## Part 0: Background — 为什么量化 KV Cache

### 问题：KV Cache 占太多显存

```
Llama-3.1-8B, 64 并发请求, 各 8K context:
  KV cache = 64 × 8192 × 128KB/token = 64 GB
  几乎吃满一张 A100 (80GB)

如果能把 KV cache 压缩到一半甚至四分之一:
  INT8: 64 GB → 32 GB    ← 可以服务 128 个并发
  INT4: 64 GB → 16 GB    ← 可以服务 256 个并发
```

### 量化的基本思想

把 FP16 (16 bit) 的 KV 数据压缩成 INT8 (8 bit) 或 INT4 (4 bit)：

```
FP16:  每个数 2 bytes, 能表示 -65504 到 +65504, 精度高
INT8:  每个数 1 byte,  只能表示 -128 到 +127,    精度低
INT4:  每个数 0.5 byte, 只能表示 -8 到 +7,        精度很低

FP16 值:  [0.123, -0.456, 0.789, -0.012, ...]
                    ↓ 量化
INT8 值:  [16, -58, 101, -2, ...]
                    ↓ 反量化（近似恢复）
FP16 近似: [0.125, -0.453, 0.789, -0.016, ...]   ← 有误差但接近
```

### 量化公式

```
量化:    q = round(x / scale)           FP16 → INT8
反量化:  x̂ = q × scale                  INT8 → FP16 (近似)

scale = max(|x|) / 127                 使 INT8 的范围覆盖数据范围
```

### 不同量化粒度

| 粒度 | scale 怎么算 | 精度 | 存储开销 |
|------|-------------|------|---------|
| Per-tensor | 整个 tensor 一个 scale | 最低 | 1 个 float |
| Per-channel | 每个 head 一个 scale | 中等 | num_heads 个 float |
| Per-token | 每个 token 一个 scale | 较高 | seq_len 个 float |
| Grouped | 每 group_size 个元素一个 scale | 最高 | 多个 float |

---

<a id="part-1-quantization-math"></a>
## Part 1: 基础量化数学 [热身]

> **文件**: `vkv/engine/quantizer.py`  
> **测试**: `uv run pytest tests/test_phase3.py -k "part1" -v`

### Task 1.1: 实现 `compute_scale()`

给定一个 FP16 tensor，计算量化 scale：

```python
scale = max(|x|) / (2^(bits-1) - 1)

# INT8: scale = max(|x|) / 127
# INT4: scale = max(|x|) / 7
```

### Task 1.2: 实现 `quantize_tensor()`

用 scale 把 FP16 tensor 量化成 INT8：

```python
q = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
```

### Task 1.3: 实现 `dequantize_tensor()`

反量化回 FP16：

```python
x_approx = q.float() * scale
```

---

<a id="part-2-per-tensor"></a>
## Part 2: Per-tensor INT8 Quantizer [核心]

> **文件**: `vkv/engine/quantizer.py` 的 `PerTensorQuantizer`  
> **测试**: `uv run pytest tests/test_phase3.py -k "part2" -v`

### Background

最简单的量化方式：整个 tensor 共用一个 scale。

```
输入: key tensor [num_kv_heads, seq_len, head_dim] = [8, 100, 128]
      总共 102400 个数字 → 找最大绝对值 → 算一个 scale

优点: 简单，存储开销最小（只多存 1 个 float）
缺点: 如果某个 head 的值特别大，其他 head 的精度被浪费
```

### Task 2.1: 实现 `PerTensorQuantizer.quantize()`
### Task 2.2: 实现 `PerTensorQuantizer.dequantize()`

---

<a id="part-3-per-channel"></a>
## Part 3: Per-channel INT8 Quantizer [核心]

> **文件**: `vkv/engine/quantizer.py` 的 `PerChannelQuantizer`  
> **测试**: `uv run pytest tests/test_phase3.py -k "part3" -v`

### Background

每个 KV head 有自己的 scale，精度更高：

```
输入: [num_kv_heads, seq_len, head_dim] = [8, 100, 128]

Per-tensor: 1 个 scale → 102400 个数字共享
Per-channel: 8 个 scale → 每个 head 独立量化

head 0 的值范围: [-0.5, 0.5] → scale_0 = 0.5/127 = 0.00394
head 1 的值范围: [-2.0, 2.0] → scale_1 = 2.0/127 = 0.01575
                                ↑ 每个 head 用自己的 scale，不互相影响
```

### Task 3.1: 实现 `PerChannelQuantizer.quantize()`

scales shape: `[num_kv_heads]` — 每个 head 一个 scale。

### Task 3.2: 实现 `PerChannelQuantizer.dequantize()`

---

<a id="part-4-grouped-int4"></a>
## Part 4: Grouped INT4 Quantizer [核心]

> **文件**: `vkv/engine/quantizer.py` 的 `GroupedQuantizer`  
> **测试**: `uv run pytest tests/test_phase3.py -k "part4" -v`

### Background

INT4 只有 16 个值 (-8 到 7)，精度很低。用分组来弥补：

```
head_dim = 128, group_size = 32

每 32 个元素一组，每组有自己的 scale:
  group 0: 元素 0-31   → scale_0
  group 1: 元素 32-63  → scale_1
  group 2: 元素 64-95  → scale_2
  group 3: 元素 96-127 → scale_3

4 个 group × 每个 1 个 float scale = 4 个 float 开销
换来 4× 压缩（FP16 → INT4）
```

### Task 4.1: 实现 `GroupedQuantizer.quantize()`
### Task 4.2: 实现 `GroupedQuantizer.dequantize()`

INT4 的特殊处理：PyTorch 没有 int4 类型，用 int8 存，范围限制在 [-8, 7]。

---

<a id="part-5-integration"></a>
## Part 5: 量化 KV Cache 集成 [核心]

> **文件**: `vkv/engine/quantizer.py` 的 `QuantizedCacheManager`  
> **测试**: `uv run pytest tests/test_phase3.py -k "part5" -v`

### Background

将量化器集成到 BlockManager 的写入/读取流程中：

```
普通流程:
  write_kv(key_fp16) → gpu_key_cache 存 FP16

量化流程:
  write_kv(key_fp16) → 量化成 INT8 → gpu_key_cache 存 INT8 (省一半空间)
  read_kv() → 读 INT8 → 反量化成 FP16 → 返回
```

### Task 5.1: 实现 `QuantizedCacheManager.write_quantized()`

写入时量化：

```python
def write_quantized(self, block_id, layer_idx, slot_idx, key, value):
    q_key, k_scale = self.quantizer.quantize(key)
    q_value, v_scale = self.quantizer.quantize(value)
    # 存量化后的数据和 scale
```

### Task 5.2: 实现 `QuantizedCacheManager.read_dequantized()`

读取时反量化：

```python
def read_dequantized(self, block_id, layer_idx, slot_idx):
    q_key, k_scale = # 读取量化数据和 scale
    key = self.quantizer.dequantize(q_key, k_scale)
    # 返回近似 FP16 数据
```

---

<a id="part-6-evaluation"></a>
## Part 6: 量化精度评估 [进阶]

> **文件**: `vkv/engine/quantizer.py`  
> **测试**: `uv run pytest tests/test_phase3.py -k "part6" -v`

### Task 6.1: 实现 `compute_quantization_error()`

衡量量化带来的误差：

```python
def compute_quantization_error(original, reconstructed):
    mse = ((original - reconstructed) ** 2).mean()
    cosine_sim = F.cosine_similarity(original.flatten(), reconstructed.flatten(), dim=0)
    return mse, cosine_sim
```

### Task 6.2: 对比不同量化策略的精度

生成随机 KV 数据，对比 per-tensor vs per-channel vs grouped 的精度。

---

## 运行测试

```bash
uv run pytest tests/test_phase3.py -v
uv run pytest tests/test_phase3.py -k "part1" -v
uv run pytest tests/test_phase3.py -k "part2" -v
uv run pytest tests/test_phase3.py -k "part3" -v
uv run pytest tests/test_phase3.py -k "part4" -v
uv run pytest tests/test_phase3.py -k "part5" -v
uv run pytest tests/test_phase3.py -k "part6" -v
```
