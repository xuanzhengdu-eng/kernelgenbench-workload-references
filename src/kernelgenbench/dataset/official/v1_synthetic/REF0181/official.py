# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/layers.py#L64-L72
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/utils.py#L110-L154
import torch
from deepspeed.inference.quantization.utils import DeQuantizer


def run(x, quantized_weight, quant_scale, quant_min, bias, num_bits, group_size,
        group_dim):
    dequantizer = DeQuantizer(
        {
            "num_bits": num_bits,
            "group_size": group_size,
            "group_dim": group_dim,
            "symmetric": False,
        },
        x.dtype,
    )
    weight = dequantizer.dequantize(quantized_weight, quant_scale, quant_min)
    return torch._C._nn.linear(x, weight, bias)
