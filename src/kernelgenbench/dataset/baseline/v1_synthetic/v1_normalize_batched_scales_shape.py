"""vLLM MoE scale-shape organisation, batched entry.

Official source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/utils.py#L682-L695

Contract:
  * ``scales`` is None, a scalar tensor, a 2-D tensor or an already 3-D tensor;
    ``num_experts`` is a required positive int.
  * None returns None; a 3-D input is returned unchanged (same object).
  * A one-element tensor is replicated with ``repeat_interleave`` into
    (num_experts, 1, 1); this path allocates, so it does not alias the input.
  * Any other tensor is reshaped to a view of (num_experts, -1, last-dim); an
    invalid element count raises RuntimeError from ``view`` exactly like
    upstream. num_experts itself is not validated upstream, so neither is it
    here.

The optional ``num_experts`` argument of the legacy reference is a different
signature from the non-batched entry, so the two are separate Items.
"""
import torch


def v1_normalize_batched_scales_shape(scales, num_experts):
    if scales is not None and scales.ndim < 3:
        if scales.numel() == 1:
            scales = scales.view(1)
            scales = torch.repeat_interleave(scales, num_experts, dim=0).view(
                num_experts, 1, 1)
        else:
            scales = scales.view(num_experts, -1, scales.size(-1))
    return scales
