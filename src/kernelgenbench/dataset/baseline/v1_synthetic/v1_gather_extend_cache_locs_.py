"""SGLang extend/verify cache-location gather, source-faithful reference.

https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/speculative/cache_locs.py#L679-L749
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/spec.cpp#L618-L674

The output is allocated here, with exactly ``batch_size * draft_token_num``
int64 entries; each request's ``[start, end)`` range is copied to the running
prefix-sum offset.  Only the first ``sum(end - start)`` entries are written, so
a caller that sizes the buffer at the maximum capacity leaves an untouched
tail.  Returns the freshly allocated dense buffer; inputs are unchanged.
"""
import torch


def v1_gather_extend_cache_locs_(req_pool_indices, req_to_token, start_offset, end_offset,
                                 batch_size, draft_token_num, device):
    assert req_to_token.dtype == torch.int32 and req_to_token.ndim == 2
    assert req_pool_indices.ndim == start_offset.ndim == end_offset.ndim == 1
    assert start_offset.dtype == end_offset.dtype
    assert start_offset.numel() == end_offset.numel() == batch_size
    out = torch.empty((batch_size * draft_token_num,), dtype=torch.int64, device=device)
    position = 0
    for row in range(batch_size):
        req = int(req_pool_indices[row])
        start = int(start_offset[row])
        end = int(end_offset[row])
        assert 0 <= req < req_to_token.shape[0]
        assert 0 <= start <= end <= req_to_token.shape[1]
        count = end - start
        assert position + count <= out.numel()
        out[position:position + count].copy_(req_to_token[req, start:end])
        position += count
    return out
