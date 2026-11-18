"""DeepSpeed gather_rows: row gather with zero fill for out-of-range indices.

Official source (deepspeedai/DeepSpeed, commit
8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_kernels.py#L256-L308

``gather_rows`` is the Triton forward kernel of ``permute_rows``. ``generate_
permute_indices`` marks padding slots with -1, so an index outside
``[0, src.shape[0])`` produces an all-zero row instead of reading memory.
``src`` is 2-D and ``index`` a 1-D integer tensor on the same device; the kernel
addresses both linearly, so it copies the (contiguous) source and allocates a
fresh row-major output. ``src`` and ``index`` are not written.
"""
import torch


def v1_gather_rows_zero_oob(src, index):
    assert src.ndim == 2 and index.ndim == 1
    assert index.dtype in (torch.int32, torch.int64)
    src = src.contiguous()
    index = index.contiguous()
    out = torch.empty(
        (index.numel(), src.shape[1]), dtype=src.dtype, device=src.device
    )
    if out.numel() == 0:
        return out
    rows = index.to(torch.int64)
    valid = (rows >= 0) & (rows < src.shape[0])
    out[valid] = src[rows[valid]]
    out[~valid] = 0
    return out
