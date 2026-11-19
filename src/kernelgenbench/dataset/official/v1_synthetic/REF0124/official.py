# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/router.py#L230-L262
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L181-L203
from megatron.core.transformer.moe.router import TopKRouter

run = TopKRouter.sinkhorn_load_balancing
