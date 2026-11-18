"""vLLM SiluAndMulWithClamp, gate clamped before the sigmoid, for OP-562393570774.

Class (forward_native / forward_cuda, alpha/beta are construction-time state):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/activation.py#L195-L243

Caller-visible contract: ``out = clamp(x[:d], max=limit) * sigmoid(alpha *
clamp(x[:d], max=limit)) * (clamp(x[d:], -limit, limit) + beta)``.  The clamp is
applied to the gate *before* the sigmoid and before the multiply.

This is not the same entry as ``SwigluStepAndMul`` (v1_clamped_swiglu_vllm_post),
which computes ``clamp(silu(gate), max=limit) * clamp(up)``: clamping after the
activation.  The two disagree for any gate whose silu exceeds the limit, which
is the evidence recorded by verify/row42_split_evidence_vllm.py.
"""
import torch
import torch.nn.functional as F  # noqa: F401  (kept for parity with the source module)


def v1_clamped_swiglu_vllm_pre(y, clamp_value, alpha=1.0, beta=0.0):
    """Packed ``[..., 2H]``; gate clamped before the sigmoid, value clamped symmetrically."""
    if y.shape[-1] % 2 or y.shape[-1] == 0:
        raise ValueError("vLLM expects packed gate and value halves")
    gate, value = y.chunk(2, dim=-1)
    gate = gate.clamp(max=clamp_value)
    value = value.clamp(min=-clamp_value, max=clamp_value)
    return gate * torch.sigmoid(alpha * gate) * (value + beta)
