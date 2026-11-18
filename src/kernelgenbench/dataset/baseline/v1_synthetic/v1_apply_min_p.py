"""vLLM minimum-probability filtering on the log scale, in place.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/min_p.py#L8-L59

The Triton ``_min_p_kernel`` path used by the V1 GPU worker:

* ``logits`` is ``[num_tokens, vocab]``, rows may be strided; ``min_p`` is
  ``[max_num_reqs]`` float32 and ``expanded_idx_mapping`` maps each row.
* ``min_p == 0`` leaves the row untouched (kernel early return).
* Otherwise ``threshold = max(logits) + log(min_p)`` and every entry strictly
  below it becomes ``-inf``; ties at the threshold survive.
* Mutates ``logits`` in place and returns ``None``.
"""
import torch


def v1_apply_min_p(logits, expanded_idx_mapping, min_p):
    p = min_p[expanded_idx_mapping.long()].float().unsqueeze(1)
    threshold = logits.float().amax(dim=-1, keepdim=True) + torch.log(p)
    invalid = (p != 0) & (logits.float() < threshold)
    logits.masked_fill_(invalid, float("-inf"))
    return None
