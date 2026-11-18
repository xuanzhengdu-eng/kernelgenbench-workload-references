"""DeepSpeed AutoEP per-row weighting (forward only, no top-k reduction).

Official source:
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/autoep_fused_token_ops.py#L450-L454
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/autoep_fused_token_ops.py#L383-L408

Contract (``fused_row_weighting`` on CUDA with Triton, exercised by autograd):
- ``rows`` is ``[N, H]`` contiguous fp16/bf16 CUDA; ``weights`` is ``[N, 1]``
  contiguous fp32 on the same device.
- Output is ``(rows.float() * weights).to(rows.dtype)``, a fresh tensor of
  ``rows`` dtype; the fp32 multiply happens before the single rounding point.
- ``N == 0`` or ``H == 0`` is legal and returns an empty/zero-sized result.
- Inputs are not modified and the output never aliases them. The official
  wrapper additionally asserts CUDA device, dtype, shape and contiguity.
"""
import torch


def v1_moe_row_weight(rows, weights):
    assert rows.ndim == 2 and rows.is_contiguous() and rows.dtype in (torch.float16, torch.bfloat16)
    assert weights.shape == (rows.shape[0], 1) and weights.dtype == torch.float32
    assert weights.is_contiguous() and rows.device == weights.device
    return (rows.float() * weights).to(rows.dtype)
