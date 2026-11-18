"""DeepSpeed EP padded row permutation (gather with zero padding rows).

Official source:
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/ep_kernels.py#L362-L387

Contract (single-rank forward of ``deepspeed.moe.ep_kernels.permute_rows`` on
CUDA with Triton):
- ``rows`` is a 2-D CUDA tensor ``[N, H]``; ``permuted_indices`` is a 1-D
  int32 tensor on the same device. A complete permutation of ``0..N-1`` plus
  padding entries (negative or ``>= N``) is what ``generate_permute_indices``
  produces; the forward itself accepts any index and treats every out-of-range
  entry as a padding slot.
- Output is a fresh ``[len(permuted_indices), H]`` tensor in ``rows.dtype``;
  valid slots copy the selected row, padding slots are exactly zero.
- Neither input is modified and the output never aliases ``rows``.
"""
import torch


def v1_ds_permute_rows(rows, permuted_indices):
    assert rows.dim() == 2
    assert permuted_indices.dim() == 1
    index = permuted_indices.to(torch.int64)
    valid = (index >= 0) & (index < rows.shape[0])
    if rows.shape[0] == 0 or index.numel() == 0:
        return torch.zeros(
            (index.numel(), rows.shape[1]), dtype=rows.dtype, device=rows.device
        )
    safe = torch.where(valid, index, torch.zeros_like(index))
    gathered = rows.index_select(0, safe)
    return torch.where(valid.unsqueeze(1), gathered, torch.zeros_like(gathered))
