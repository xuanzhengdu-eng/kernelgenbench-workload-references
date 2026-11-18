"""NVIDIA Megatron-LM row-parallel linear (official pinned commit).

Official source (Megatron-LM 266f1c97):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/layers.py#L1072-L1286

Contract (differs from the vLLM/SGLang siblings, therefore a separate item):
  * `input_is_parallel=False` scatters the last dimension; sequence parallel
    requires `input_is_parallel=True`.
  * the matmul bias is never fused; after the (all-)reduce the bias is added
    on every rank, or returned separately when `skip_bias_add=True`.
  * `sequence_parallel=True` performs reduce-scatter on dim 0 and keeps the
    token shard of this rank.
  * explicit expert communication skips the reduction entirely and requires
    `skip_bias_add=True`.
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist


def v1_megatron_row_parallel_linear(x, weight, bias, group, input_is_parallel,
                                    sequence_parallel, skip_bias_add,
                                    explicit_expert_comm):
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if input_is_parallel:
        local = x
    else:
        assert not sequence_parallel
        assert x.shape[-1] == weight.shape[-1] * world
        local = x.chunk(world, dim=-1)[rank].contiguous()
    out = F.linear(local, weight, None)
    if explicit_expert_comm:
        assert skip_bias_add
    elif sequence_parallel:
        assert out.shape[0] % world == 0
        shard = torch.empty(out.shape[0] // world, *out.shape[1:],
                            dtype=out.dtype, device=out.device)
        dist.reduce_scatter_tensor(shard, out.contiguous(), group=group)
        out = shard
    else:
        dist.all_reduce(out, group=group)
    if skip_bias_add:
        return out, bias
    return (out + bias if bias is not None else out), None
