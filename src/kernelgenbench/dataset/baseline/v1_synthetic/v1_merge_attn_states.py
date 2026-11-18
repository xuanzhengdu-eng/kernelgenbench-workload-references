"""vLLM non-FP8 split-attention merge into caller-owned buffers.

``merge_attn_states`` combines prefix/suffix partial attention outputs using
their log-sum-exp values for the first ``prefill_tokens_with_context`` tokens
and copies the suffix output for the remaining tokens.  ``output`` and the
optional ``output_lse`` are mutated in place; the function returns None and
does not modify the prefix/suffix inputs.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/ops/triton_merge_attn_states.py#L14-L214
"""
import torch


def v1_merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                         suffix_lse, output_lse=None,
                         prefill_tokens_with_context=None):
    n = output.shape[0] if prefill_tokens_with_context is None else prefill_tokens_with_context
    p = prefix_lse[:, :n].float()
    s = suffix_lse[:, :n].float()
    p = torch.where(p == float("inf"), float("-inf"), p)
    s = torch.where(s == float("inf"), float("-inf"), s)
    maximum = torch.maximum(p, s)
    empty = maximum == float("-inf")
    safe_max = torch.where(empty, torch.zeros_like(maximum), maximum)
    ep = torch.exp(p - safe_max)
    es = torch.exp(s - safe_max)
    den = ep + es
    safe_den = torch.where(empty, torch.ones_like(den), den)
    merged = (prefix_output[:n].float() * (ep / safe_den).T.unsqueeze(-1)
              + suffix_output[:n].float() * (es / safe_den).T.unsqueeze(-1))
    output[:n].copy_(merged.to(output.dtype))
    output[n:].copy_(suffix_output[n:])
    if output_lse is not None:
        output_lse[:, :n].copy_(
            torch.where(empty, float("-inf"), safe_max + torch.log(safe_den)))
        output_lse[:, n:].copy_(suffix_lse[:, n:])
    return None
