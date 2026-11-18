"""Accuracy and performance for v1_sgl_renorm_top_p.

Official source (pinned commit 62384de1ae5e3bbd3159afd1639525b0bf39fbf9):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/sampling_renorm.py#L56-L70

Contract: ``probs`` is a nonnegative ``[B, V]`` tensor.  Each row is sorted
descending; entries whose strictly-preceding cumulative mass exceeds ``top_ps``
are zeroed (the largest entry always survives), survivors are renormalized and
scattered back to the original columns.  The output is freshly allocated and
the input is unchanged.  The FlashInfer kernel takes float32 contiguous rows, so
this item only uses contiguous inputs; the strided contract stays with the
PyTorch top-p item.

Item: v1_sgl_renorm_top_p (723-row 585, top-p FlashInfer branch).
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows582_591_checks as checks


# B, V, per-row top-p, whether the rows come from a wider strided buffer.
CASES = [(3, 8, (0.5, 0.9, 1.0), False), (3, 17, (0.0, 0.4, 0.8), False)]
# FlashInfer's top-p kernel takes float32 probabilities and returns float32,
# which is what SGLang's sampler passes; fp16/bf16 probs are not a supported
# contract of this kernel (it silently upcasts and returns float32).
COMBOS = [(case, torch.float32) for case in CASES]

TOLERANCE = {
    torch.float32: (1e-6, 1e-7),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}


def input_build(config):
    """probs ``[B, V]`` normalized rows; top_ps float32 ``[B]`` in [0, 1].

    The strided case is a ``[B, V+3]`` buffer sliced to ``[:, :V]``, so the
    row stride is V+3 while the column stride stays 1.
    """
    (batch, vocab, limits, strided), dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    probs = torch.rand((batch, vocab), dtype=dtype, generator=generator)
    probs = probs / probs.sum(-1, keepdim=True)
    if strided:
        storage = torch.zeros((batch, vocab + 3), dtype=dtype)
        storage[:, :vocab] = probs
        probs = storage[:, :vocab]
    return {"probs": probs, "top_ps": torch.tensor(limits, dtype=torch.float32)}


def compute_ref(probs, top_ps):
    return kernelgenbench.baseline.v1_sgl_renorm_top_p_fast(probs, top_ps)


def compute_res(probs, top_ps):
    return kernelgenbench.solution.v1_sgl_renorm_top_p_fast(probs, top_ps)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == torch.float32
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


def compute_ref_performance(probs, top_ps):
    return checks.measure(compute_ref, {"probs": probs, "top_ps": top_ps}, probs.device)


def compute_res_performance(probs, top_ps):
    return checks.measure(compute_res, {"probs": probs, "top_ps": top_ps}, probs.device)


@label("v1_sgl_renorm_top_p_fast")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_renorm_top_p_fast(config):
    (batch, vocab, limits, strided), dtype = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    before = ref_inputs["probs"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    assert res_out.shape == (batch, vocab)
    assert not torch._C._is_alias_of(res_out, res_inputs["probs"])
    # Survivors are renormalized to a probability distribution.
    torch.testing.assert_close(res_out.sum(-1), torch.ones(batch, dtype=dtype,
                                                          device=res_out.device),
                               rtol=2e-2 if dtype is torch.bfloat16 else 2e-3,
                               atol=2e-2 if dtype is torch.bfloat16 else 1e-3)
    # Inputs are unchanged, including the strided buffer.
    torch.testing.assert_close(ref_inputs["probs"], before, rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["probs"], before, rtol=0, atol=0)
    assert res_inputs["probs"].stride() == ref_inputs["probs"].stride()

    # A row with top_p=0 keeps exactly one entry.
    zero_rows = [i for i, limit in enumerate(limits) if limit == 0.0]
    for row in zero_rows:
        assert torch.count_nonzero(res_out[row]) == 1

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return checks.result(ref_time, res_time)
