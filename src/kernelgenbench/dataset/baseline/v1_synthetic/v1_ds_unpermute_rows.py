# Source-checked synthetic-workload benchmark reference.
"""OP-c4159a95fc8b: DeepSpeed unpermute_rows forward.

Official source:
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_kernels.py#L379-L387
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_kernels.py#L312-L330

Contract: `permuted_indices` is a complete one-to-one permutation of
0 .. num_rows - 1; every other entry is padding (negative or >= num_rows).
The result is the [num_rows, hidden] original order, padding rows are never
read. This is the Triton/CUDA autograd function `_UnpermuteRows`.
"""
import torch


def v1_ds_unpermute_rows(rows, permuted_indices, num_rows):
    assert rows.dim() == 2
    ids = permuted_indices.to(torch.int64)
    valid = (ids >= 0) & (ids < int(num_rows))
    inverse = torch.empty(int(num_rows), dtype=torch.int64, device=ids.device)
    inverse[ids[valid]] = torch.arange(ids.numel(), device=ids.device, dtype=torch.int64)[valid]
    return rows.index_select(0, inverse)
