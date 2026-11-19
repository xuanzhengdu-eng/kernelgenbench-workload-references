# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L1339-L1373
from megatron.core.transformer.moe.moe_utils import RouterGatingLinearFunction

run = RouterGatingLinearFunction.backward
