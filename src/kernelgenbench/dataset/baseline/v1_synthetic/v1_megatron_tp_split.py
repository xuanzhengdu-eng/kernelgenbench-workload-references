"""Megatron rank-local split along the first dimension.

Official source (NVIDIA/Megatron-LM, pinned commit 266f1c97):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/tensor_parallel/mappings.py#L56-L77

Scalar-input sub-contract of ``_split_along_first_dim``: the process group is
replaced by explicit ``world_size``/``rank`` scalars, exactly the two values the
official helper reads through ``group.size()`` / ``group.rank()``.  Behaviour
kept identical:
  * ``world_size == 1`` returns the input object unchanged (official bypass),
  * the first dimension must be divisible by ``world_size`` (official assert),
  * the result is this rank's contiguous slice (official ``.contiguous()``).
No communication happens on this path.
"""


def v1_megatron_tp_split(input_, world_size, rank):
    if world_size == 1:
        return input_
    assert 0 <= rank < world_size
    assert input_.shape[0] % world_size == 0, (
        "First dimension of the tensor should be divisible by tensor parallel size"
    )
    n = input_.shape[0] // world_size
    return input_.narrow(0, rank * n, n).contiguous()
