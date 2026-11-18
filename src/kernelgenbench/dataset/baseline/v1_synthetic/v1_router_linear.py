"""Megatron RouterGatingLinearFunction forward (router dtype projection).

Official source (NVIDIA Megatron-LM, pinned commit):
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/transformer/moe/moe_utils.py#L1299-L1336

Contract:
  * ``inp`` is ``[..., H]``, ``weight`` is ``[O, H]``, ``bias`` is ``[O]`` or
    ``None``, and ``router_dtype`` is the dtype used for the GEMM.
  * The projection is ``inp.to(router_dtype) @ weight.to(router_dtype).T`` plus
    ``bias.to(router_dtype)`` when present; the result keeps ``router_dtype``
    and the input's leading shape: ``[..., O]``.
  * Inputs are not modified; the output is a fresh tensor.  The official
    ``te_general_gemm`` path is used for every router dtype except float64.
"""
import torch
import torch.nn.functional as F


def v1_router_linear(inp, weight, bias, router_dtype):
    assert inp.shape[-1] == weight.shape[1]
    assert weight.ndim == 2
    if bias is not None:
        assert bias.shape == (weight.shape[0],)
    return F.linear(inp.to(router_dtype), weight.to(router_dtype),
                    None if bias is None else bias.to(router_dtype))
