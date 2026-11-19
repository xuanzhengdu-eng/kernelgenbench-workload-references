# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/layers.py#L70-L77
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/utils.py#L110-L146
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/quantization/quantize_intX.cu#L195-L235
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/quantization/pt_binding.cpp#L83-L101
import torch

from deepspeed.inference.quantization.utils import DeQuantizer


def run(dequantizer, x, quantized_weight, quant_scale, quant_min, bias):
    weight = dequantizer.dequantize(quantized_weight.view(torch.uint8), quant_scale, quant_min)
    return torch._C._nn.linear(x, weight, bias)


__all__ = ["run", "DeQuantizer"]
