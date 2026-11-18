"""Accuracy and performance for v1_einsum_sec_sm_ecm (723-row 97).

Reference: DeepSpeed ``deepspeed.moe.sharded_moe.einsum`` restricted to the
``sec,sm->ecm`` rule used by the MoE capacity dispatch.
Official source (pinned commit):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/moe/sharded_moe.py#L117-L149

The candidate side calls that official function and flips the module-level
``USE_EINSUM`` switch per case, so both official bodies (torch.einsum and the
matmul rewrite) are exercised.  ``a`` is [S,E,C] and ``b`` is [S,M]; the output
is [E,C,M].  The strided layout puts the contraction axis ``s`` at stride 2,
which the rewrite must copy through ``reshape``.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows92_101_checks as checks


TOLERANCE = {
    torch.float32: (4e-6, 2e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}

# (S, E, C, M): capacity x experts x per-expert channels contracted with m rows.
SHAPES = [(1, 2, 4, 3), (3, 2, 4, 5), (7, 4, 8, 2), (64, 8, 16, 128),
          (2, 1, 1, 1)]

COMBOS = [
    (shape, dtype, use_einsum, layout)
    for shape in SHAPES
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for use_einsum in (True, False)
    for layout in ("contiguous", "strided")
    if layout == "contiguous" or shape[0] > 1
]


def input_build(config):
    shape, dtype, use_einsum, layout = config
    s, e, c, m = shape
    generator = torch.Generator(device="cpu").manual_seed(20261118)
    a = torch.randn((s, e, c), dtype=dtype, generator=generator) * 0.5
    b = torch.randn((s, m), dtype=dtype, generator=generator) * 0.5
    if layout == "strided":
        buffer = torch.zeros((s, e, c * 2), dtype=dtype)
        buffer[..., ::2] = a
        a = buffer[..., ::2]
    return {"rule": "sec,sm->ecm", "a": a, "b": b, "use_einsum": use_einsum}


def compute_ref(rule, a, b, use_einsum):
    return kernelgenbench.baseline.v1_einsum_sec_sm_ecm(rule, a, b, use_einsum)


def compute_res(rule, a, b, use_einsum):
    return kernelgenbench.solution.v1_einsum_sec_sm_ecm(rule, a, b, use_einsum)


def compare(ref_out, res_out, dtype):
    checks.assert_tensor(res_out, ref_out, dtype, *TOLERANCE[dtype])


def compute_ref_performance(rule, a, b, use_einsum):
    return checks.measure(kernelgenbench.baseline.v1_einsum_sec_sm_ecm,
                          {"rule": rule, "a": a, "b": b,
                           "use_einsum": use_einsum}, a.device)


def compute_res_performance(rule, a, b, use_einsum):
    return checks.measure(kernelgenbench.solution.v1_einsum_sec_sm_ecm,
                          {"rule": rule, "a": a, "b": b,
                           "use_einsum": use_einsum}, a.device)


@label("v1_einsum_sec_sm_ecm")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_einsum_sec_sm_ecm(config):
    shape, dtype, use_einsum, layout = config
    s, e, c, m = shape
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    ref_a, ref_b = ref_inputs["a"].clone(), ref_inputs["b"].clone()
    res_a, res_b = res_inputs["a"].clone(), res_inputs["b"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    assert tuple(res_out.shape) == (e, c, m)
    assert res_out.stride() == ref_out.stride(), "layout of the result changed"
    assert torch.equal(res_inputs["a"], res_a), "solution modified a"
    assert torch.equal(res_inputs["b"], res_b), "solution modified b"
    assert torch.equal(ref_inputs["a"], ref_a), "reference modified a"
    assert torch.equal(ref_inputs["b"], ref_b), "reference modified b"

    # Timing uses the device tensors; the CPU dict above only holds sources.
    return checks.timings(compute_ref_performance(**ref_inputs),
                          compute_res_performance(**res_inputs))
