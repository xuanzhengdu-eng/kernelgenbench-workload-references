# Source-checked synthetic-workload benchmark reference.
"""OP-f60d05da0a08: DeepSpeed permute_rows backward (explicit gradient helper).

Official source:
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_kernels.py#L332-L345
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_kernels.py#L312-L330

`_PermuteRows.backward` returns `(_UnpermuteRows.apply(grad_out, permuted_indices,
inverse), None, None)`, i.e. `rows.index_select(0, inverse)`. This reference is
the explicit helper for that backward: `rows` is the incoming gradient of the
padded, expert-major buffer ([num_permuted_rows, hidden]) and the result is
[num_rows, hidden]. The two trailing None values mirror the backward signature.
"""
import torch


def v1_ds_permute_rows_backward(rows, permuted_indices, num_rows):
    assert rows.dim() == 2
    ids = permuted_indices.to(torch.int64)
    valid = (ids >= 0) & (ids < int(num_rows))
    inverse = torch.empty(int(num_rows), dtype=torch.int64, device=ids.device)
    inverse[ids[valid]] = torch.arange(ids.numel(), device=ids.device, dtype=torch.int64)[valid]
    out = rows.index_select(0, inverse)
    return out, None, None
