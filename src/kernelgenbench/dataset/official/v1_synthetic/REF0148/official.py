# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_softmax.py#L60-L106
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/fusions/fused_softmax.py#L299-L350
from megatron.core.fusions.fused_softmax import (
    FusedScaleMaskSoftmax,
    ScaledMaskedSoftmax,
    ScaledSoftmax,
    ScaledUpperTriangMaskedSoftmax,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import attention_mask_func

OFFICIAL_ENTRY = (
    "megatron.core.fusions.fused_softmax.FusedScaleMaskSoftmax.forward_torch_softmax "
    "+ torch.autograd (VJP at the pinned ScaledMaskedSoftmax.backward site)"
)
OFFICIAL_COMMIT = "266f1c97dca477ca8b92c16087506da2000b0b84"
OFFICIAL_FUSED_BACKWARD = "scaled_masked_softmax_cuda.backward / scaled_upper_triang_masked_softmax_cuda.backward"

__all__ = [
    "FusedScaleMaskSoftmax",
    "ScaledMaskedSoftmax",
    "ScaledSoftmax",
    "ScaledUpperTriangMaskedSoftmax",
    "AttnMaskType",
    "attention_mask_func",
    "OFFICIAL_ENTRY",
    "OFFICIAL_COMMIT",
    "OFFICIAL_FUSED_BACKWARD",
]
