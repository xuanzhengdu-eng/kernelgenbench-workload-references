# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation.cpp#L38-L68
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation.py#L21-L62
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation_cuda.cu#L16-L20
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation_cuda.cu#L50-L89
import torch

from deepspeed.inference.v2.inference_utils import ActivationType, DtypeEnum
from deepspeed.inference.v2.kernels.core_ops.bias_activations.bias_activation import (
    CUDABiasActivation,
)

OFFICIAL_ACT_FN = ActivationType.IDENTITY


def official_bias_activation(channels, dtype=torch.float16):

    return CUDABiasActivation(channels, DtypeEnum(dtype), OFFICIAL_ACT_FN)
