"""NVIDIA Megatron fused symmetric-memory RS + residual add + RMSNorm + AG.

Official source (Megatron-LM 266f1c97):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/communication/torch_symm_triton/fused_collectives.py#L210-L280

The official entry is `fused_multimem_rs_add_norm_ag(residual_output_tensor,
input_tensor, symm_mem_hdl, residual_input_tensor, rms_norm_weights, eps)`: it
reduce-scatters the symmetric-memory input across the TP group, adds the
residual shard in place, normalises with RMSNorm and all-gathers the result
back into the symmetric buffer. This reference keeps the matmul that produces
the partial input outside the fused call (the fused kernel starts at the
collective), matching the cited line range.
"""
import torch
import torch.distributed as dist


def v1_megatron_row_nvls_fused_(x, weight, residual, symmetric_buffer, gamma, eps,
                                tp_group=None):
    world = dist.get_world_size(tp_group)
    rank = dist.get_rank(tp_group)
    symmetric_buffer.copy_(torch.matmul(x.float(), weight.float().t()).to(torch.bfloat16))
    partial = symmetric_buffer.float()
    dist.all_reduce(partial, group=tp_group)
    rs = partial.to(torch.bfloat16).chunk(world, 0)[rank].contiguous()
    residual.copy_((rs.float() + residual.float()).to(torch.bfloat16))
    v = residual.float()
    norm = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
            * gamma.float()).to(torch.bfloat16)
    chunks = [torch.empty_like(norm) for _ in range(world)]
    dist.all_gather(chunks, norm, group=tp_group)
    symmetric_buffer.copy_(torch.cat(chunks, 0))
    return residual
