"""Accuracy and performance for v1_cpu_adam_update_.

Operator: DeepSpeed scalar CPU Adam/AdamW single dense-tensor update
(``ds_adam_step`` -> ``Adam_Optimizer::Step_1``).
Official source (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adam/cpu_adam_impl.cpp#L35-L87
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adam/cpu_adam_impl.cpp#L251-L277

Contract: contiguous CPU param/grad/exp_avg/exp_avg_sq with equal shape;
param and grad share one dtype (fp32/fp16/bf16); both moments are fp32 or the
param dtype.  FP32 math with optional bias correction:
``g = wd*param + grad`` for coupled Adam (``wd > 0`` and ``adamw == False``);
``m = b1*m + (1-b1)*g``; ``v = b2*v + (1-b2)*g*g``;
``c1 = 1-b1**step``, ``c2 = 1/sqrt(1-b2**step)`` when bias correction is on;
``update = m/(sqrt(v)*c2 + eps) * (-lr/c1)``;
AdamW adds ``-lr*wd*param`` before the update.  The call mutates param, exp_avg
and exp_avg_sq, leaves grad unchanged, and returns status 0.  Empty tensors are
a no-op.

Item: v1_cpu_adam_update_ (723-row 150).  ``compute_ref`` loads the reference
from ``kernelgenbench.baseline``; ``compute_res`` calls whatever the verifier
bound as ``kernelgenbench.solution`` for the same label, i.e. the real pinned
DeepSpeed CPU extension during verification.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows142_151_checks as checks


# (shape, step, lr, weight_decay, adamw, bias_correction)
CASES = [
    ((9,), 3, 0.03, 0.1, False, True),
    ((4, 7), 7, 0.01, 0.02, True, True),
    ((5,), 4, 0.02, 0.03, True, False),
    ((1,), 1, 0.01, 0.0, True, True),
    ((0,), 1, 0.1, 0.0, False, False),
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
    """param/grad/m/v: shape in dtype; moments fp32 or dtype; all CPU."""
    (shape, step, lr, weight_decay, adamw, bias_correction), dtype, state_fp32 = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    state_dtype = torch.float32 if state_fp32 else dtype
    return {
        "param": torch.randn(shape, generator=generator).to(dtype),
        "grad": torch.randn(shape, generator=generator).to(dtype),
        "exp_avg": torch.randn(shape, generator=generator).to(state_dtype),
        "exp_avg_sq": (0.1 + torch.rand(shape, generator=generator)).to(state_dtype),
        "step": step,
        "lr": lr,
        "beta1": 0.9,
        "beta2": 0.99,
        "eps": 1e-6,
        "weight_decay": weight_decay,
        "adamw": adamw,
        "bias_correction": bias_correction,
    }


def compute_ref(param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps,
                weight_decay, adamw, bias_correction):
    return kernelgenbench.baseline.v1_cpu_adam_update_(
        param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps,
        weight_decay, adamw, bias_correction)


def compute_res(param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps,
                weight_decay, adamw, bias_correction):
    return kernelgenbench.solution.v1_cpu_adam_update_(
        param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps,
        weight_decay, adamw, bias_correction)


def compare(ref_out, res_out, config, inputs, before):
    (shape, step, lr, weight_decay, adamw, bias_correction), dtype, state_fp32 = config
    assert ref_out == res_out == 0, "the C++ entry returns status 0"
    for name in ("param", "grad", "exp_avg", "exp_avg_sq"):
        assert inputs[name].shape == shape
        assert inputs[name].device.type == "cpu"
        assert inputs[name].data_ptr() == before["ptrs"][name], f"{name} was reallocated"
    assert inputs["param"].dtype == inputs["grad"].dtype == dtype
    assert inputs["exp_avg"].dtype == inputs["exp_avg_sq"].dtype == \
        (torch.float32 if state_fp32 else dtype)
    assert torch.equal(inputs["grad"], before["grad"]), "Adam keeps grad unchanged"

    beta1, beta2 = inputs["beta1"], inputs["beta2"]
    p = before["param"].float()
    g = before["grad"].float()
    m = before["exp_avg"].float()
    v = before["exp_avg_sq"].float()
    if weight_decay > 0 and not adamw:
        g = p * weight_decay + g
    m_new = m * beta1 + g * (1 - beta1)
    v_new = v * beta2 + g * g * (1 - beta2)
    if bias_correction:
        c1 = 1 - beta1 ** step
        c2 = 1 / torch.sqrt(torch.tensor(1 - beta2 ** step, dtype=torch.float32))
    else:
        c1 = 1.0
        c2 = 1.0
    update = m_new / (torch.sqrt(v_new) * c2 + 1e-6) * (-lr / c1)
    p_new = p
    if weight_decay > 0 and adamw:
        p_new = p + (-lr * weight_decay) * p
    expected = {"param": p_new + update, "exp_avg": m_new, "exp_avg_sq": v_new}
    rtol, atol = TOLERANCE[dtype]
    for name, value in expected.items():
        torch.testing.assert_close(inputs[name].float(), value, rtol=rtol, atol=atol)

    if shape and shape[0] and lr > 0:
        assert not torch.equal(inputs["param"], before["param"])


def compute_ref_performance(**kwargs):
    return checks.measure(compute_ref, kwargs, torch.device("cpu"))


def compute_res_performance(**kwargs):
    return checks.measure(compute_res, kwargs, torch.device("cpu"))


@label("v1_cpu_adam_update_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_cpu_adam_update_(config):
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs)
    res_inputs = checks.fresh(inputs)
    before = checks.fresh(inputs)
    before["ptrs"] = {name: res_inputs[name].data_ptr()
                      for name in ("param", "grad", "exp_avg", "exp_avg_sq")}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config, res_inputs, before)

    return checks.timings(compute_ref_performance(**inputs),
                          compute_res_performance(**inputs))
