# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/contexts/fused_kv_append_kernel.py#L93-L187
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/contexts/fused_kv_append_kernel.py#L21-L90
from megatron.core.inference.contexts.fused_kv_append_kernel import (
    triton_append_key_value_cache as run,
)
