# NVIDIA Megatron-LM pinned fused collective: multimem reduce-scatter +
# residual add + RMSNorm + all-gather, written into the symmetric-memory buffer.
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/communication/torch_symm_triton/fused_collectives.py#L210-L280
# https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/inference_layers.py#L420-L435
"""Reference semantics for the Megatron multimem RS+residual+RMS+AG collective.

Every rank holds one local slice of the partial input in a symmetric-memory
buffer of shape ``(world * local_tokens, hidden)``.  The collective sums the
partials across the group, keeps the rank-local slice, adds the residual input,
writes that residual sum into ``residual_output_tensor``, applies RMSNorm with
``rms_norm_weights`` and publishes every rank's normalized slice into the same
symmetric buffer, so the buffer holds the all-gathered result in rank order.

This is a plain-torch model of the pinned NVIDIA entry; the real GPU path is the
multimem Triton kernel bound by the paired test.
"""
import torch
import torch.distributed as dist

def v1_megatron_rs_add_rms_ag_(residual_output_tensor,input_tensor,residual_input_tensor,rms_norm_weights,eps,group):
    world=dist.get_world_size(group);rank=dist.get_rank(group)
    assert input_tensor.dtype==residual_input_tensor.dtype==residual_output_tensor.dtype==rms_norm_weights.dtype==torch.bfloat16
    assert input_tensor.ndim==2 and input_tensor.shape[0]%world==0 and input_tensor.shape[1]%8==0
    full=input_tensor.float().clone();dist.all_reduce(full,group=group)
    local=full.to(torch.bfloat16).chunk(world,0)[rank]
    result=(local+residual_input_tensor).to(torch.bfloat16)
    residual_output_tensor.copy_(result)
    z=result.float();norm=(z*torch.rsqrt(z.square().mean(dim=-1,keepdim=True)+eps)*rms_norm_weights.float()).to(torch.bfloat16)
    parts=[torch.empty_like(norm) for _ in range(world)];dist.all_gather(parts,norm.contiguous(),group=group)
    input_tensor.copy_(torch.cat(parts,dim=0))
    return residual_output_tensor
