"""vLLM minimum-probability filtering on the probability scale.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/logits_processor/builtin.py#L102-L115

``MinPLogitsProcessor.apply`` is the standalone logits-processor backend. It is
selected when the batch is processed by ``vllm/v1/sample`` instead of the GPU
worker kernel, and its contract differs from the log-scale kernel:

* Softmax over the vocabulary, then ``max_probability * min_p`` per row.
* Every entry whose probability is below that product becomes ``-inf``.
* Mutates ``logits`` and returns the same tensor object (not ``None``).
* ``min_p == 0`` rows pass through unchanged only because the caller disables
  the processor; this function itself still applies the threshold (which is 0).
"""
import torch


def v1_apply_min_p_probability(logits, expanded_idx_mapping, min_p):
    p = min_p[expanded_idx_mapping.long()].float().unsqueeze(1)
    probability_values = torch.nn.functional.softmax(logits, dim=-1)
    max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
    adjusted_min_p = max_probabilities.mul_(p)
    invalid_token_mask = probability_values < adjusted_min_p
    logits.masked_fill_(invalid_token_mask, float("-inf"))
    return logits
