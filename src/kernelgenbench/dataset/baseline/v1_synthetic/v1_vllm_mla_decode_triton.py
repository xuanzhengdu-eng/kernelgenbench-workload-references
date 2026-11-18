"""Single-token vLLM Triton MLA forward_mqa with a 512-wide value prefix.

The test fixture supplies each request's block table and sequence length from
the public MLA decode metadata. The module and layer have unquantized weights.
"""
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/backends/mla/triton_mla.py#L246-L345
import torch


def v1_vllm_mla_decode_triton(query, kv_cache, scale, block_tables, seq_lens):
    if isinstance(query, tuple):
        query = torch.cat(query, dim=-1)
    outputs, logsumexp = [], []
    for batch, length in enumerate(seq_lens.tolist()):
        positions = torch.arange(length, device=query.device)
        page_size = kv_cache.shape[1]
        pages = block_tables[batch, positions // page_size].long()
        keys = kv_cache[pages, positions % page_size].float()
        scores = (query[batch].float() @ keys.T) * scale
        outputs.append((scores.softmax(-1) @ keys[:, :512]).to(query.dtype))
        logsumexp.append(scores.logsumexp(-1).to(query.dtype))
    return torch.stack(outputs), torch.stack(logsumexp)
