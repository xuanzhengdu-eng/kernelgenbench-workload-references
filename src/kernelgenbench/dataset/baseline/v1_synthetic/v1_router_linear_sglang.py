"""SGLang unquantized linear (router/index projection fallback).

Official source (SGLang, pinned commit):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/quantization/unquant.py#L793-L839

Contract:
  * ``inp`` is ``[..., K]`` and ``weight`` is ``[N, K]``; ``bias`` is ``[N]`` or
    ``None``.
  * The result is ``F.linear(inp, weight, bias)`` in the input dtype, with the
    input's leading shape: ``[..., N]``.  The official method may dispatch the
    same contract to an AMX, AITER, or BF16 GEMM backend; the fallback and every
    backend share these semantics.
  * Inputs are not modified; the output is a fresh tensor.
"""
import torch
import torch.nn.functional as F


def v1_router_linear_sglang(inp, weight, bias):
    assert inp.shape[-1] == weight.shape[1]
    if bias is not None:
        assert bias.shape == (weight.shape[0],)
    return F.linear(inp, weight, bias)
