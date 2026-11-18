"""DeepSpeed Triton grouped GEMM, per-group concatenated token rows.

Official source (DeepSpeed, pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/triton_ops/group_gemm_triton.py#L462-L489

Contract:
  * ``mat_a`` is ``[M, K]`` (row-major; non-contiguous input is allowed and is
    compacted by the official kernel).
  * ``mat_b`` is contiguous ``[E, K, N]`` when ``trans_b=False`` (out = a @ b)
    or ``[E, N, K]`` when ``trans_b=True`` (out = a @ b^T).
  * ``offs`` is the int32 cumulative row boundary ``[E]`` with
    ``offs[-1] == M``; empty groups are allowed.
  * The result is ``[M, N]`` in the input dtype; matmuls accumulate in fp32 and
    are cast back once per group, matching the Triton kernel's numerics.
  * Inputs are not modified; there is no in-place write and no bias.
"""
import torch


def v1_ds_grouped_gemm(mat_a, mat_b, offs, trans_b=False):
    assert mat_a.ndim == 2 and mat_b.ndim == 3
    assert mat_a.dtype == mat_b.dtype
    assert offs.ndim == 1 and offs.numel() == mat_b.shape[0]
    assert int(offs[-1]) == mat_a.shape[0]
    chunks = []
    start = 0
    for expert, end in enumerate(offs.tolist()):
        weight = mat_b[expert].T if trans_b else mat_b[expert]
        chunk = (mat_a[start:end].float() @ weight.float()).to(mat_a.dtype)
        chunks.append(chunk)
        start = end
    return torch.cat(chunks, dim=0)
