# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/utils.py#L48-L99
from deepspeed.inference.quantization.utils import Quantizer

QuantizerEntry = Quantizer


def quantize(tensor, group_dim, group_size, num_bits):
    return QuantizerEntry(
        {"group_dim": group_dim, "group_size": group_size,
         "num_bits": num_bits, "symmetric": False}).quantize(tensor)


run = quantize
