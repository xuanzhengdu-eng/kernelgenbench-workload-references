# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/layers.py#L75-L108
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/utils.py#L96-L165
from deepspeed.inference.quantization.layers import QuantizedEmbedding
from deepspeed.inference.quantization.utils import DeQuantizer
from deepspeed.inference.quantization.utils import concat_to_compat_param

OfficialQuantizedEmbedding = QuantizedEmbedding
OfficialDeQuantizer = DeQuantizer
OfficialConcatToCompatParam = concat_to_compat_param
OfficialQuantizedEmbeddingForward = QuantizedEmbedding.forward
OfficialDeQuantize = DeQuantizer.dequantize
