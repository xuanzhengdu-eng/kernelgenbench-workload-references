"""Megatron rank-local split along the last dimension.

Official source (NVIDIA/Megatron-LM, pinned commit 266f1c97):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L36-L53

Scalar-input sub-contract of ``_split_along_last_dim``.  This is a different
official implementation from ``_split_along_first_dim``: it splits the *last*
axis through ``split_tensor_along_last_dim`` and keeps this rank's contiguous
chunk.  Kept behaviour:
  * ``world_size == 1`` returns the input object unchanged (official bypass),
  * the last dimension must split evenly into ``world_size`` pieces,
  * the result is contiguous, so a non-contiguous input is copied.
No communication happens on this path.
"""


def v1_megatron_tp_split_last(input_, world_size, rank):
    if world_size == 1:
        return input_
    assert 0 <= rank < world_size
    dim = input_.ndim - 1
    assert input_.shape[dim] % world_size == 0, (
        "Last dimension of the tensor should be divisible by tensor parallel size"
    )
    n = input_.shape[dim] // world_size
    return input_.narrow(dim, rank * n, n).contiguous()
