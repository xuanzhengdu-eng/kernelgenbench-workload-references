"""vLLM sampling temperature, in place per request with 0/1 bypass.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/gumbel.py#L22-L63

``apply_temperature`` launches ``_temperature_kernel`` with one program per
(expanded token, 8192-wide vocabulary block). The kernel:

* maps each row through ``expanded_idx_mapping`` to its request temperature;
* returns early, leaving the row bit-identical, when the temperature is 0 or 1;
* otherwise divides every logit by the temperature in fp32 and stores it back in
  the logits dtype.

Mutates ``logits`` in place and returns ``None``; the mapping/temperature tensors
are untouched.
"""
import torch


def v1_apply_temperature_(logits, expanded_idx_mapping, temperature):
    t = temperature[expanded_idx_mapping.long()].float().unsqueeze(1)
    bypass = (t == 0) | (t == 1)
    safe = torch.where(bypass, 1.0, t)
    logits.copy_((logits.float() / safe).to(logits.dtype))
    return None
