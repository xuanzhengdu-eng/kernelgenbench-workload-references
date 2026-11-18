# vLLM pinned causal-convolution chunk scheduling metadata for the unpinned-host
# path.  The CPU query starts are turned into per-BLOCK_M chunk counts, a repeated
# batch list, per-chunk offsets and two pre-padded device pointer arrays.
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/backends/utils.py#L1203-L1249
"""vLLM causal-convolution chunk scheduling metadata, unpinned-host mode."""
import torch


def v1_causal_conv_chunk_metadata(query_start_loc_p_cpu, device):
    q = query_start_loc_p_cpu
    assert q.device.type == 'cpu' and q.ndim == 1 and q.dtype in (torch.int32, torch.int64)
    lengths = q.diff()
    assert (lengths >= 0).all()
    nums = lengths // 8 + (lengths % 8 != 0).to(lengths.dtype)
    batches, offsets = [], []
    for i, n in enumerate(nums.tolist()):
        batches.extend([i] * n)
        offsets.extend(range(n))
    mlist = torch.tensor(batches, dtype=torch.int64)
    offsetlist = torch.tensor(offsets, dtype=torch.int32)
    total = len(batches)
    capacity = max(1024, total) * 2
    batch_ptr = torch.full((capacity,), -1, dtype=torch.int32, device=device)
    offset_ptr = torch.full((capacity,), -1, dtype=torch.int32, device=device)
    batch_ptr[:total].copy_(mlist)
    offset_ptr[:total].copy_(offsetlist)
    return {8: {'nums': nums, 'tot': total, 'mlist': mlist, 'mlist_len': total,
                'offsetlist': offsetlist, 'batch_ptr': batch_ptr,
                'token_chunk_offset_ptr': offset_ptr}}, batch_ptr, offset_ptr
