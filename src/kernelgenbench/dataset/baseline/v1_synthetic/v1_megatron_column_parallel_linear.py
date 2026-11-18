"""Megatron column-parallel linear forward, without sequence parallelism.

Source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/layers.py#L934-L1035

Contract: ``x`` is ``[..., input_size]`` and the stored weight is this rank's
``[output_size_per_partition, input_size]`` block.  A forward-time ``weight``
argument overrides the stored one and must have exactly that shape, otherwise
``RuntimeError``; when both are missing ``RuntimeError``.  The local output is
``F.linear(x, weight, bias)`` (no bias term when ``skip_bias_add``).  A
gather_output setting, overridden by ``runtime_gather_output`` when it is not
None, all-gathers the local outputs along the last dimension when the TP group
has more than one rank.  The return is always ``(output, bias)`` where ``bias``
is the module bias when ``skip_bias_add`` is set and ``None`` otherwise.  No
input tensor is modified.  ``group``/``output_size_per_partition``/
``input_size`` replace the module's runtime state.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F


def v1_megatron_column_parallel_linear(x, stored_weight, weight, bias, group,
                                       output_size_per_partition, input_size,
                                       gather_output, runtime_gather_output,
                                       skip_bias_add):
    if weight is None:
        if stored_weight is None:
            raise RuntimeError(
                "weight was not supplied to ColumnParallelLinear forward pass "
                "and skip_weight_param_allocation is True.")
        weight = stored_weight
    else:
        expected_shape = (output_size_per_partition, input_size)
        if tuple(weight.shape) != expected_shape:
            raise RuntimeError(
                f"supplied weight's shape is {tuple(weight.shape)}, "
                f"not {expected_shape} as expected")

    local = F.linear(x, weight, None if skip_bias_add else bias)
    gather = gather_output if runtime_gather_output is None else runtime_gather_output
    world = dist.get_world_size(group)
    if gather and world > 1:
        parts = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(parts, local.contiguous(), group=group)
        out = torch.cat(parts, dim=-1)
    else:
        out = local
    return out, (bias if skip_bias_add else None)
