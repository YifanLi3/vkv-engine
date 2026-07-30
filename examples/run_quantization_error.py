import torch
from vkv.engine.quantizer import (
    compute_quantization_error,
    PerTensorQuantizer,
    PerChannelQuantizer,
    GroupedQuantizer,
)

torch.manual_seed(42)
original = torch.randn(8, 128)

print('=' * 55)
print(f'  Original tensor shape: {list(original.shape)}')
print('=' * 55)

for name, q in [
    ('PerTensor  INT8', PerTensorQuantizer(bits=8)),
    ('PerChannel INT8', PerChannelQuantizer(bits=8)),
    ('Grouped    INT4', GroupedQuantizer(bits=4, group_size=32)),
]:
    qtensor, scale = q.quantize(original)
    reconstructed = q.dequantize(qtensor, scale)
    mse, cos_sim = compute_quantization_error(original, reconstructed)
    print(f'{name} | MSE: {mse.item():.6f} | Cosine Sim: {cos_sim.item():.6f}')