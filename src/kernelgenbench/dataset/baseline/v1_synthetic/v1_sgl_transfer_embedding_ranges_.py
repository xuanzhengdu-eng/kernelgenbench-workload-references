"""Copy host/device embedding row ranges with atomic plan validation.
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu#L822-L971
"""
def v1_sgl_transfer_embedding_ranges_(src,dst,src_starts,dst_starts,lengths):
    if src.ndim!=2 or dst.ndim!=2 or src.dtype!=dst.dtype or src.shape[1]!=dst.shape[1] or not src.is_contiguous() or not dst.is_contiguous():raise ValueError('incompatible embedding tensors')
    if not (len(src_starts)==len(dst_starts)==len(lengths)):raise ValueError('plan length mismatch')
    for s,d,n in zip(src_starts,dst_starts,lengths):
        if n<0:raise ValueError('negative length')
        if n==0:continue
        if s<0 or d<0 or s+n>src.shape[0] or d+n>dst.shape[0]:raise ValueError('range out of bounds')
    for s,d,n in zip(src_starts,dst_starts,lengths):
        if n:dst[d:d+n].copy_(src[s:s+n],non_blocking=True)
    return None
