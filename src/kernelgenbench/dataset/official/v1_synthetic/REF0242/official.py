# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/optimizer/clip_grads.py#L55-L142
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/utils.py#L1178-L1186
from megatron.core.optimizer.clip_grads import get_grad_norm_fp32 as run
