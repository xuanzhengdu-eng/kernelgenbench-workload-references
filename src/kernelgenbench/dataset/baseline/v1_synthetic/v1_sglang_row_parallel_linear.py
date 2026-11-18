"""SGLang RowParallelLinear forward (official pinned commit).

Official source (SGLang 62384de1):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/linear.py#L1749-L1839

The non-sequence-parallel floating point forward returns (output, deferred bias or None). The module fuses the bias on rank 0 and optionally all-reduces its local product.
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist


def v1_sglang_row_parallel_linear(x, weight, bias, group, input_is_parallel,
                                  skip_bias_add=False, reduce_results=True):
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if input_is_parallel:
        local = x
    else:
        assert x.shape[-1] == weight.shape[-1] * world
        local = x.chunk(world, dim=-1)[rank].contiguous()
    rank_bias = None if (rank > 0 or skip_bias_add) else bias
    out = F.linear(local, weight, rank_bias)
    if reduce_results and world > 1:
        dist.all_reduce(out, group=group)
    return out, (bias if skip_bias_add else None)
