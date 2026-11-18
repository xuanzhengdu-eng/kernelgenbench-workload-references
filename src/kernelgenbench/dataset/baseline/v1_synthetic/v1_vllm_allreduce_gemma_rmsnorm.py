"""All-reduce plus Gemma residual RMSNorm, unfused collective path.

Fused entry and its explicit all-reduce fallback:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py#L139-L190
GemmaRMSNorm (weight is an additive offset: ``x * (1 + w)``):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/layernorm.py#L145-L174
Unfused add + RMSNorm IR op:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/ir/ops/layernorm.py#L44-L62

This is the explicit collective path: the per-rank partial is all-reduced, the
residual is added, and Gemma RMSNorm is applied, all in fp32 before casting
back to the activation dtype.  It returns ``(normed, new_residual)`` as fresh
tensors and never writes ``hidden_states`` or ``residual`` in place.  At
``world_size == 1`` the collective is the identity, which is the branch the
single-GPU test exercises; the multi-rank reduction is not benchmarked here.
"""
import torch
import torch.distributed as dist


def v1_vllm_allreduce_gemma_rmsnorm(hidden_states, residual, weight, eps, group=None):
    reduced = hidden_states.clone()
    if dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.all_reduce(reduced, group=group)
    summed = reduced.float() + residual.float()
    out = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + eps) * (1 + weight.float())
    return out.to(hidden_states.dtype), summed.to(hidden_states.dtype)
