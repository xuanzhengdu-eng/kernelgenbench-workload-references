# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L735-L782
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_mla_yarn_rope_apply.py#L379-L487
from megatron.core.fusions.fused_mla_yarn_rope_apply import (
    fused_apply_mla_rope_for_kv as run,
)

__all__ = ["run"]
