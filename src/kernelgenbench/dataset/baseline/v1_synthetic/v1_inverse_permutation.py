"""DeepSpeed inverse permutation with padding slots ignored.

Official source (deepspeedai/DeepSpeed, commit
8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_kernels.py#L312-L328

``permuted_indices`` holds each of ``0 .. num_rows - 1`` exactly once; every
other entry is padding (negative or ``>= num_rows``). Each valid value is
scattered to its own position and padding goes to a distinct slot past
``num_rows`` that is sliced away, so the result is an int64 ``[num_rows]``
tensor of original-row positions. The input is not modified.
"""
import torch


def v1_inverse_permutation(permuted_indices, num_rows):
    assert permuted_indices.ndim == 1 and num_rows >= 0
    positions = torch.arange(
        permuted_indices.numel(), device=permuted_indices.device, dtype=torch.int64
    )
    indices = permuted_indices.to(torch.int64)
    valid = (indices >= 0) & (indices < num_rows)
    targets = torch.where(valid, indices, num_rows + positions)
    inverse = torch.empty(
        num_rows + permuted_indices.numel(),
        dtype=torch.int64,
        device=permuted_indices.device,
    )
    inverse.scatter_(0, targets, positions)
    return inverse[:num_rows]
