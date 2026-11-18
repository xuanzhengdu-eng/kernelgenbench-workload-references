"""SGLang CPU pre-apply of the top-k weights, then replace them with ones.

Official source:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/moe/topk.py#L1283-L1293

Contract (``apply_topk_weights_cpu``; argument order is the fixed source order
``need_apply, topk_weights, inputs``):
- ``inputs`` is ``[..., H]``, ``topk_weights`` is ``[..., 1]`` and broadcasts
  against the last dimension.
- ``need_apply=False`` returns the *same two objects* ``(inputs, topk_weights)``
  with no copy and no mutation.
- ``need_apply=True`` casts ``topk_weights`` to ``inputs.dtype`` before the
  elementwise multiply (one rounding step, unlike DeepSpeed's fp32 multiply then
  cast), and returns ``(new_inputs, torch.ones_like(topk_weights, float32))``.
  Both results are fresh tensors; the inputs themselves are not modified.
"""
import torch


def v1_moe_preapply_weights(need_apply, topk_weights, inputs):
    if not need_apply:
        return inputs, topk_weights
    return inputs * topk_weights.to(inputs.dtype), torch.ones_like(
        topk_weights, dtype=torch.float32
    )
