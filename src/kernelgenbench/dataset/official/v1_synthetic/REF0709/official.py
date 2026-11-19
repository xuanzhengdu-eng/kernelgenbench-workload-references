# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/communication/torch_symm_triton/fused_collectives.py#L210-L280
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/inference_layers.py#L420-L435
from megatron.core.inference.communication.torch_symm_triton import (
    fused_multimem_rs_add_norm_ag as run,
)
