"""vLLM in-place masking of empty DCP shard rows in the LSE tensor.

Source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/ops/dcp.py#L49-L69
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/ops/dcp.py#L72-L97

Validation runs before any device work so the function is CUDA-graph safe.
A boundary belongs to the next nonempty interval, including repeated offsets;
``searchsorted(..., right=True)`` on the trailing boundaries is the exact
selection rule.  Returns None in every path.
"""
import torch


def v1_mask_empty_dcp_lse_(lse, seq_lens, query_start_loc):
    # _validate_dcp_empty_shard_args
    if seq_lens is None and query_start_loc is None:
        return None
    if seq_lens is None or query_start_loc is None:
        raise ValueError("seq_lens and query_start_loc must be provided together")
    if (
        seq_lens.ndim != 1
        or query_start_loc.ndim != 1
        or query_start_loc.shape[0] != seq_lens.shape[0] + 1
    ):
        raise ValueError(
            "query_start_loc must contain one boundary per sequence"
        )

    # A rank can receive no local sequences during CUDA-graph warmup.
    if seq_lens.shape[0] == 0:
        lse.fill_(float("-inf"))
        return None

    row_indices = torch.arange(
        lse.shape[0], device=lse.device, dtype=query_start_loc.dtype
    )
    sequence_indices = torch.searchsorted(
        query_start_loc[1:], row_indices, right=True
    ).clamp_max(seq_lens.shape[0] - 1)
    empty_rows = (row_indices >= query_start_loc[-1]) | (
        seq_lens[sequence_indices] == 0
    )
    lse.masked_fill_(empty_rows[:, None], float("-inf"))
    return None
