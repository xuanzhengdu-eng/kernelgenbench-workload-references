"""Fused vLLM repetition, frequency and presence penalties with draft history.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/penalties.py#L107-L217

The kernel rewrites ``logits`` in place, one row per draft-expanded token:

* ``prompt_bin_mask[req]`` is a packed bitmask, bit ``v % 32`` of word
  ``v // 32`` saying token ``v`` appeared in the prompt.
* ``output_bin_counts[req]`` counts committed output tokens; draft tokens of the
  same request are added by comparing ``token_ids`` over the previous
  ``expanded_local_pos`` positions.
* Present tokens (prompt bit or count > 0) get the repetition scale, positive
  logits divided and the rest multiplied, then ``frequency_penalty * count`` and
  ``presence_penalty`` are subtracted.
* Returns ``None``; only ``logits`` changes.
"""
import torch


def v1_fused_sampling_penalties_(
    logits,
    expanded_idx_mapping,
    token_ids,
    expanded_local_pos,
    repetition_penalty,
    frequency_penalty,
    presence_penalty,
    prompt_bin_mask,
    output_bin_counts,
):
    rows, vocab = logits.shape
    device = logits.device
    req = expanded_idx_mapping.long()
    v = torch.arange(vocab, device=device)
    r = torch.arange(rows, device=device)
    packed = prompt_bin_mask[req[:, None], (v // 32)[None, :]].to(torch.int64)
    prompt = ((packed >> (v % 32)[None, :]) & 1).bool()
    counts = output_bin_counts[req].to(torch.int32)
    previous = (r[None, :] >= (r - expanded_local_pos + 1)[:, None]) & (
        r[None, :] <= r[:, None]
    )
    counts = counts + (
        ((token_ids[None, :, None] == v[None, None, :]) & previous[:, :, None])
        .sum(1)
        .to(torch.int32)
    )
    observed = counts > 0
    rep = repetition_penalty[req, None].float()
    scale = torch.where(prompt | observed, rep, 1.0)
    values = logits.float() * torch.where(logits > 0, 1.0 / scale, scale)
    values = values - frequency_penalty[req, None].float() * counts
    values = values - presence_penalty[req, None].float() * observed
    logits.copy_(values.to(logits.dtype))
    return None
