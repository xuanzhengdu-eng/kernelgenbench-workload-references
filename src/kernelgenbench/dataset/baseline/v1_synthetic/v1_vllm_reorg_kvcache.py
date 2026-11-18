"""vLLM MLA chunked-context KV cache reorder and unpad.

Official implementation:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/attention/mla_attention.py#L2801-L2872

Contract: the all-gathered cache is rank-major with ``toks`` rows per rank and
each chunk starts at the same source offset on every rank.  For each request
chunk (``padded_local_chunk_seq_lens_lst``) and each rank
(``local_context_lens_allranks``), copy
``min(max(0, length - local_start), cap)`` rows starting at
``rank * toks + src_token_idx`` into the request-major output.  ``sum_seq_len``
is the total copied row count and ``max_seq_len`` the largest per-request
count; both are asserted.  Two fresh tensors are returned, the inputs are not
modified.
"""
import torch


def v1_vllm_reorg_kvcache(
    allgatered_kv_c_normed,
    allgatered_k_pe,
    padded_local_chunk_seq_lens_lst,
    local_context_lens_allranks,
    local_starts,
    sum_seq_len,
    max_seq_len,
    toks,
):
    indices = []
    start = 0
    max_seen = 0
    for cap, lengths, local_start in zip(
        padded_local_chunk_seq_lens_lst, local_context_lens_allranks, local_starts
    ):
        count = 0
        for rank, length in enumerate(lengths):
            size = min(max(0, length - local_start), cap)
            indices.extend(range(rank * toks + start, rank * toks + start + size))
            count += size
        max_seen = max(max_seen, count)
        start += cap
    assert len(indices) == sum_seq_len and max_seen == max_seq_len
    idx = torch.tensor(indices, device=allgatered_kv_c_normed.device, dtype=torch.long)
    return allgatered_kv_c_normed.index_select(0, idx), allgatered_k_pe.index_select(0, idx)
