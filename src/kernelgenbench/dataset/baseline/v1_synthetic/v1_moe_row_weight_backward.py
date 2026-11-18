"""Explicit DeepSpeed AutoEP row-weighting backward.

Official source:
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/autoep_fused_token_ops.py#L410-L447

``_FusedRowWeighting.backward`` is reached through autograd on
``fused_row_weighting(rows, weights)`` (line 450). Contract for the tested
sub-domain:
- ``rows`` is ``[N, H]`` contiguous fp16/bf16 CUDA; ``weights`` is ``[N, 1]``
  contiguous fp32 with ``requires_grad``; ``grad_output`` has ``rows`` shape and
  dtype and may be non-contiguous (the source makes it contiguous).
- Returns ``(grad_rows, grad_weights)`` where
  ``grad_rows = (grad_output.float() * weights).to(rows.dtype)`` and
  ``grad_weights = sum(grad_output.float() * rows.float(), dim=1, keepdim=True)``
  in fp32 (the kernel reduces fp32 partial blocks).
- ``N == 0`` yields empty gradients; ``H == 0`` yields zero weight gradients.
- Neither input is modified and both gradients are fresh tensors. Autograd may
  report ``None`` for a gradient of an empty leaf; the wrapper in the test
  exposes that as an empty tensor.
"""
import torch


def v1_moe_row_weight_backward(rows, weights, grad_output):
    assert rows.ndim == 2 and rows.is_contiguous() and rows.dtype in (torch.float16, torch.bfloat16)
    assert weights.shape == (rows.shape[0], 1) and weights.dtype == torch.float32
    assert weights.is_contiguous()
    assert grad_output.shape == rows.shape and grad_output.dtype == rows.dtype
    assert rows.device == weights.device == grad_output.device
    return (
        (grad_output.float() * weights).to(rows.dtype),
        (grad_output.float() * rows.float()).sum(dim=1, keepdim=True),
    )
