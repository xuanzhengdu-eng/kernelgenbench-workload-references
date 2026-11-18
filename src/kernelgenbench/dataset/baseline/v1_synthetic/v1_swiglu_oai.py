"""vLLM GPT-OSS SwiGLU-OAI (interleaved gate/value).

Official source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/activation.py#L486-L493

Contract: ``x`` has last dimension ``2H`` and interleaved pairs
``(gate, up) = (x[..., ::2], x[..., 1::2])``.

* gate is clamped only from above: ``min(gate, limit)``;
* up is clamped symmetrically: ``clamp(up, -limit, limit)``;
* output is ``(up + 1) * gate * sigmoid(alpha * gate)`` with shape
  ``x.shape[:-1] + (H,)`` and the input dtype.

This differs from SGLang's Phi3 ``gegelu`` on infinities and pair layout, so the
two are separate Operator Items.
"""
import torch


def v1_swiglu_oai(x, alpha, limit):
    gate = x[..., ::2].clamp(min=None, max=limit)
    up = x[..., 1::2].clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu
