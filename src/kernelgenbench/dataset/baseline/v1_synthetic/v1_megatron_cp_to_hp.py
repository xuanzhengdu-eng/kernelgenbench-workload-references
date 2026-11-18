"""Reference: Megatron context-parallel to head-parallel all-to-all (3-D).

Official implementation (pinned Megatron-LM commit
``266f1c97dca477ca8b92c16087506da2000b0b84``):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/ssm/mamba_context_parallel.py#L304-L339

``_all_to_all_cp2hp(input_, cp_group)`` takes a 3-D tensor
``[sequence / cp_size, batch, local_hidden]`` and returns
``[sequence, batch, local_hidden / cp_size]``:

1. flatten the first two dimensions to ``[s * b, h]``;
2. split ``h`` into ``cp_size`` chunks and concatenate them along dim 0;
3. one equal-split all-to-all over the CP group;
4. reshape back to ``[s * cp_size, b, h / cp_size]``.

Contract: the input must be 3-D, ``h`` must be divisible by the group size, and
the result is a fresh contiguous tensor; the input is only read.  The official
function asserts the 3-D shape.
"""
import torch
import torch.distributed as dist


def v1_megatron_cp_to_hp(group, input_):
    assert input_.dim() == 3, "all_to_all_cp2hp assumes 3-d input shape"
    s_in, b_in, h_in = input_.shape
    world = group.size()
    assert h_in % world == 0, (h_in, world)
    flat = input_.reshape(-1, h_in)
    chunks = torch.split(flat, h_in // world, dim=1)
    joined = torch.cat(chunks, dim=0)
    exchanged = torch.empty_like(joined)
    dist.all_to_all_single(exchanged, joined.contiguous(), group=group)
    return exchanged.reshape(s_in * world, b_in, h_in // world)
