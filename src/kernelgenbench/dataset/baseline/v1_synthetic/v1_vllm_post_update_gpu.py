"""vLLM GPU sampling post-update, mutates request state tensors.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/input_batch.py#L566-L685

Pinned entry ``post_update`` (Triton ``_post_update_kernel``).  For each request
row with a non-negative state index, append the sampled tokens at the request's
old total length, count them into the output histogram, update the last sampled
token and total length, and adjust ``num_computed_tokens`` by
``query_len - num_rejected``.  GPU contract: CUDA tensors, one row per request
state index (no duplicates, otherwise the per-row programs race) and a flat
contiguous ``last_sampled_tokens``.  Returns None.
"""
import torch


def v1_vllm_post_update_gpu(idx_mapping, num_computed_tokens, last_sampled_tokens,
                            output_bin_counts, sampled_tokens, num_sampled, num_rejected,
                            query_start_loc, all_token_ids, total_len):
    assert idx_mapping.device.type == "cuda"
    assert last_sampled_tokens.dim() == 1 and last_sampled_tokens.is_contiguous()
    live = idx_mapping[idx_mapping >= 0].tolist()
    assert len(live) == len(set(live)), "GPU post_update requires unique request state ids"
    for row, req in enumerate(idx_mapping.tolist()):
        if req < 0:
            continue
        count = int(num_sampled[row])
        old = int(total_len[req])
        if count:
            last_sampled_tokens[req] = sampled_tokens[row, count - 1]
            total_len[req] = old + count
        for offset in range(count):
            token = int(sampled_tokens[row, offset])
            all_token_ids[req, old + offset] = token
            if output_bin_counts is not None:
                output_bin_counts[req, token] += 1
        query = 0 if query_start_loc is None else (
            int(query_start_loc[row + 1]) - int(query_start_loc[row]))
        num_computed_tokens[req] += query - int(num_rejected[row])
