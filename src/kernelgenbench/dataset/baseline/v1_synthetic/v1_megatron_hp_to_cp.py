"""Reference: Megatron head-parallel to context-parallel all-to-all (3-D).

Official implementation (pinned Megatron-LM commit
``266f1c97dca477ca8b92c16087506da2000b0b84``):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/ssm/mamba_context_parallel.py#L343-L376

``_all_to_all_hp2cp(input_, cp_group)`` is the inverse transform: it takes
``[sequence, batch, local_hidden / cp_size]``, flattens the first two
dimensions, runs one equal-split all-to-all, then splits the exchanged rows into
``cp_size`` blocks, concatenates them along the hidden dimension and restores
``[sequence / cp_size, batch, local_hidden]``.

Contract: 3-D input, ``sequence`` divisible by the group size, fresh contiguous
result, input only read.
"""
import torch
import torch.distributed as dist


def v1_megatron_hp_to_cp(group, input_):
    assert input_.dim() == 3, "all_to_all_hp2cp assumes 3-d input shape"
    s_in, b_in, h_in = input_.shape
    world = group.size()
    assert s_in % world == 0, (s_in, world)
    flat = input_.reshape(-1, h_in)
    exchanged = torch.empty_like(flat)
    dist.all_to_all_single(exchanged, flat.contiguous(), group=group)
    s_out = s_in // world
    chunks = torch.split(exchanged, s_out * b_in, dim=0)
    joined = torch.cat(chunks, dim=-1)
    return joined.reshape(s_out, b_in, h_in * world)
