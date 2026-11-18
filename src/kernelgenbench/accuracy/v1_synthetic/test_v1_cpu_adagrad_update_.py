"""Accuracy and performance for v1_cpu_adagrad_update_.

Operator: DeepSpeed CPU Adagrad single dense-tensor update
(``ds_adagrad_step`` -> ``Adagrad_Optimizer::Step_1``).
Official source (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adagrad/cpu_adagrad.cpp#L20-L55
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adagrad/cpu_adagrad.cpp#L178-L199

Contract: contiguous CPU param/grad/state with equal shape; param and grad share
one dtype (fp32/fp16/bf16), state is fp32 or the param dtype.  FP32 math:
``g = grad + wd*param`` when ``wd > 0``; ``v += g*g``;
``update = -lr*g/(sqrt(v)+eps)``.  The call mutates param and state and returns
C++ status 0.  On the pinned AVX512 build the vectorized prefix of ``grad`` is
left unchanged; only the scalar tail (``numel % 16`` elements) is replaced by
the update term.  Empty tensors are a no-op.  Optimizer object setup, the sparse
branch and noncontiguous inputs are out of scope.

Item: v1_cpu_adagrad_update_ (723-row 149).  ``compute_ref`` loads the reference
from ``kernelgenbench.baseline``; ``compute_res`` calls whatever the verifier
bound as ``kernelgenbench.solution`` for the same label, i.e. the real pinned
DeepSpeed CPU extension during verification.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows142_151_checks as checks


# (shape, step, lr, eps, weight_decay)
CASES = [
    ((9,), 3, 0.03, 1e-5, 0.1),
    ((4, 7), 7, 0.01, 1e-8, 0.02),
    ((1,), 1, 0.1, 1e-6, 0.0),
    ((0,), 1, 0.1, 1e-6, 0.0),
]

COMBOS = [
    (case, dtype, state_fp32)
    for case in CASES
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for state_fp32 in (False, True)
]

TOLERANCE = {
    torch.float32: (1e-5, 1e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}


def input_build(config):
    """param/grad: shape in dtype; state: shape in fp32 or dtype; all CPU."""
    (shape, step, lr, eps, weight_decay), dtype, state_fp32 = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    state_dtype = torch.float32 if state_fp32 else dtype
    return {
        "param": torch.randn(shape, generator=generator).to(dtype),
        "grad": torch.randn(shape, generator=generator).to(dtype),
        "state": torch.rand(shape, generator=generator).to(state_dtype),
        "step": step,
        "lr": lr,
        "eps": eps,
        "weight_decay": weight_decay,
    }


def compute_ref(param, grad, state, step, lr, eps, weight_decay):
    return kernelgenbench.baseline.v1_cpu_adagrad_update_(
        param, grad, state, step, lr, eps, weight_decay)


def compute_res(param, grad, state, step, lr, eps, weight_decay):
    return kernelgenbench.solution.v1_cpu_adagrad_update_(
        param, grad, state, step, lr, eps, weight_decay)


def compare(ref_out, res_out, config, inputs, before):
    (shape, _step, lr, eps, weight_decay), dtype, state_fp32 = config
    assert ref_out == res_out == 0, "the C++ entry returns status 0"
    for name in ("param", "grad", "state"):
        assert inputs[name].shape == shape
        assert inputs[name].device.type == "cpu"
        assert inputs[name].data_ptr() == before["ptrs"][name], f"{name} was reallocated"
    assert inputs["param"].dtype == inputs["grad"].dtype == dtype
    assert inputs["state"].dtype == (torch.float32 if state_fp32 else dtype)

    # Exact contract, evaluated in fp32 from the original tensors.
    p = before["param"].float()
    g = before["grad"].float()
    v = before["state"].float()
    if weight_decay > 0:
        g = p * weight_decay + g
    v_new = v + g * g
    update = (-lr) * (g / (torch.sqrt(v_new) + eps))
    expected = {"param": p + update, "state": v_new}
    rtol, atol = TOLERANCE[dtype]
    for name, value in expected.items():
        assert inputs[name].dtype == before[name].dtype
        torch.testing.assert_close(inputs[name].float(), value, rtol=rtol, atol=atol)

    # AVX512 split: the vectorized prefix keeps grad, only the scalar tail
    # (numel % 16 elements) is replaced by the update term.
    rounded = (p.numel() // 16) * 16
    flat_got = inputs["grad"].view(-1)
    flat_want = before["grad"].view(-1)
    flat_update = update.view(-1)
    if rounded:
        assert torch.equal(flat_got[:rounded], flat_want[:rounded]), \
            "the AVX512 prefix must leave grad unchanged"
    if p.numel() > rounded:
        torch.testing.assert_close(flat_got[rounded:].float(), flat_update[rounded:],
                                   rtol=rtol, atol=atol)


def compute_ref_performance(**kwargs):
    return checks.measure(compute_ref, kwargs, torch.device("cpu"))


def compute_res_performance(**kwargs):
    return checks.measure(compute_res, kwargs, torch.device("cpu"))


@label("v1_cpu_adagrad_update_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_cpu_adagrad_update_(config):
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs)
    res_inputs = checks.fresh(inputs)
    before = checks.fresh(inputs)
    before["ptrs"] = {name: res_inputs[name].data_ptr()
                      for name in ("param", "grad", "state")}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, res_inputs, before)

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
