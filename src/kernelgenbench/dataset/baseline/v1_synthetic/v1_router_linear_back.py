"""Megatron RouterGatingLinearFunction backward.

Official source (NVIDIA Megatron-LM, pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L1339-L1373

Contract:
  * ``inp`` is ``[..., H]``, ``weight`` is ``[O, H]``, ``bias`` is ``[O]`` or
    ``None``, and ``grad_output`` has shape ``[..., O]``.
  * ``grad_input = (grad_output @ weight.to(router_dtype)).to(inp.dtype)``,
    ``grad_weight = (grad_output.T @ inp.to(router_dtype)).to(weight.dtype)``,
    and ``grad_bias = grad_output.sum(0).to(weight.dtype)`` when bias exists,
    else ``None``.
  * Returns ``(grad_input, grad_weight, grad_bias, None)``; the last ``None``
    is the non-tensor ``router_dtype`` argument.
  * Inputs are not modified.  The official ``te_general_gemm`` path is used for
    every router dtype except float64; float64 falls back to torch.mm.
"""
import torch


def v1_router_linear_back(grad_output, inp, weight, bias, router_dtype):
    assert grad_output.shape[:-1] == inp.shape[:-1]
    assert grad_output.shape[-1] == weight.shape[0]
    assert inp.shape[-1] == weight.shape[1]
    if bias is not None:
        assert bias.shape == (weight.shape[0],)
    g = grad_output.reshape(-1, grad_output.shape[-1])
    a = inp.reshape(-1, inp.shape[-1])
    grad_input = torch.mm(g, weight.to(router_dtype)).to(inp.dtype).reshape(inp.shape)
    grad_weight = torch.mm(g.t(), a.to(router_dtype)).to(weight.dtype)
    grad_bias = g.sum(dim=0).to(weight.dtype) if bias is not None else None
    return grad_input, grad_weight, grad_bias, None
