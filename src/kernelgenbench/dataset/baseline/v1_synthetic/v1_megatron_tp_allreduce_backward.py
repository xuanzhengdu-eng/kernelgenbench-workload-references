"""Explicit backward of Megatron tensor-parallel all-reduce: pass the gradient through.

Source (pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L217-L245

``_ReduceFromModelParallelRegion.backward(ctx, grad_output)`` returns
``(grad_output, None)``: the all-reduce in the forward direction has a
pass-through backward, with no collective, allocation, dtype conversion or
numerical transformation.  Any shape, strides and floating dtype are accepted;
the returned gradient object is the caller's object, not a copy.  The second
tuple element is the gradient for the non-tensor ``group`` argument and is
always ``None``.
"""


def v1_megatron_tp_allreduce_backward(grad_output):
    return grad_output, None
