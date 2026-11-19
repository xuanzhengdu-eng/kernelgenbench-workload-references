# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation.py#L21-L62
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation_cuda.cu#L36-L87
# https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/v2/kernels/core_ops/bias_activations/bias_activation.cpp#L38-L68
from deepspeed.inference.v2.kernels.core_ops.bias_activations import CUDABiasActivation as run
