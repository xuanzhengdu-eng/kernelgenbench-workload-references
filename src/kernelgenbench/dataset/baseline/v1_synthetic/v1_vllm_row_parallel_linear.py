"""vLLM RowParallelLinear forward (official pinned commit).

Official source (vLLM f358f057):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/linear.py#L1832-L1859

Contract: the bias is fused into the GEMM on TP rank 0 only and therefore
added before the all-reduce, so it is counted exactly once. Sequence
parallelism and explicit expert communication are not part of this entry.
`reduce_results=False` leaves the partial product local; `return_bias=False`
returns only the tensor.
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist


def v1_vllm_row_parallel_linear(x, weight, bias, group, input_is_parallel,
                                skip_bias_add=False, reduce_results=True,
                                return_bias=True):
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
    if not return_bias:
        return out
    return out, (bias if skip_bias_add else None)
