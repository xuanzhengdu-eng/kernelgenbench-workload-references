"""DeepSpeed ZenFlow combined loss unscale and optional gradient clipping.

https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/runtime/zenflow/engine_stage3.py#L528-L537

The upstream method is ``unscale_and_clip_grads(self, sub_group_id, total_norm,
now_state)`` and multiplies one fp32 flat gradient buffer in place.  This
reference keeps that arithmetic and the None return for the same single-buffer
scope; ``total_norm`` is the scaled tensor norm.
"""
import torch

def v1_zenflow_unscale_clip_(grad, total_norm, loss_scale, clip_grad):
    combined_scale = loss_scale
    if clip_grad > 0.:
        clip = ((total_norm / loss_scale) + 1e-6) / clip_grad
        clip = torch.clamp(clip, min=1.0)
        combined_scale = clip * loss_scale
    grad.mul_(1. / combined_scale)
