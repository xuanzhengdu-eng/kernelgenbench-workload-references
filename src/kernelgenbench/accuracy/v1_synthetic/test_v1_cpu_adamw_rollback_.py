"""Accuracy and performance for v1_cpu_adamw_rollback_.

Operator: DeepSpeed CPU AdamW rollback single dense-tensor update
(``ds_adam_rollback`` -> ``adamw_rollback_inplace``).
Official source (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adam/cpu_adam_impl.cpp#L279-L325
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/adam/cpu_adam_impl.cpp#L327-L373

Contract: all four tensors are contiguous CPU float32 for the supported path;
any non-fp32 tensor returns -1 without mutation.  The rollback always applies
bias correction (the ``bias_correction`` argument is ignored):
``update = lr*(m/c1)/(sqrt(v/c2)+eps)``;
``param = (param+update)/(1-lr*wd)`` with non-finite results replaced by zero;
``m = (m-(1-b1)*g)/b1``; ``v = (v-(1-b2)*g*g)/b2``.  grad is preserved.
The internal step counter is decremented but is not externally observable.

Item: v1_cpu_adamw_rollback_ (723-row 151).  ``compute_ref`` loads the
reference from ``kernelgenbench.baseline``; ``compute_res`` calls whatever the
verifier bound as ``kernelgenbench.solution`` for the same label, i.e. the real
pinned DeepSpeed CPU extension during verification.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows142_151_checks as checks


# (shape, step, lr, weight_decay, bias_correction (ignored), negative_variance, dtype)
CASES = [
    ((7,), 1, 0.01, 0.01, True, False, torch.float32),
    ((4, 5), 3, 0.03, 0.1, False, False, torch.float32),
    ((3,), 4, 0.5, 2.0, True, False, torch.float32),   # lr*wd == 1 -> rollback to zero
    ((5,), 2, 0.01, 0.0, False, True, torch.float32),  # negative variance
    ((3,), 1, 0.01, 0.0, True, False, torch.float16),  # unsupported dtype -> -1
    ((0,), 1, 0.01, 0.0, True, False, torch.float32),
]

COMBOS = CASES


def input_build(config):
    """param/grad/m/v all in the config dtype; negative_variance flips v[0]."""
    shape, _step, _lr, _weight_decay, _bias_correction, negative, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    param = torch.randn(shape, generator=generator).to(dtype)
    grad = torch.randn(shape, generator=generator).to(dtype)
    exp_avg = torch.randn(shape, generator=generator).to(dtype)
    exp_avg_sq = (0.1 + torch.rand(shape, generator=generator)).to(dtype)
    if negative:
        exp_avg_sq.view(-1)[0] = -0.1
    return {
        "param": param,
        "grad": grad,
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
        "step": _step,
        "lr": _lr,
        "beta1": 0.9,
        "beta2": 0.99,
        "eps": 1e-6,
        "weight_decay": _weight_decay,
        "bias_correction": _bias_correction,
    }


def compute_ref(param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps,
                weight_decay, bias_correction):
    return kernelgenbench.baseline.v1_cpu_adamw_rollback_(
        param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps,
        weight_decay, bias_correction)


def compute_res(param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps,
                weight_decay, bias_correction):
    return kernelgenbench.solution.v1_cpu_adamw_rollback_(
        param, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps,
        weight_decay, bias_correction)


def compare(ref_out, res_out, config, inputs, before):
    shape, step, lr, weight_decay, _bias_correction, _negative, dtype = config
    assert ref_out == res_out, "reference and candidate must return the same status"
    for name in ("param", "grad", "exp_avg", "exp_avg_sq"):
        assert inputs[name].data_ptr() == before["ptrs"][name], f"{name} was reallocated"
        assert inputs[name].shape == shape
        assert inputs[name].device.type == "cpu"

    if dtype != torch.float32:
        assert ref_out == -1
        for name in ("param", "grad", "exp_avg", "exp_avg_sq"):
            assert torch.equal(inputs[name], before[name]), f"{name} changed on -1"
        return

    assert ref_out == 0
    assert torch.equal(inputs["grad"], before["grad"]), "rollback keeps grad unchanged"
    p = before["param"].float()
    g = before["grad"].float()
    m = before["exp_avg"].float()
    v = before["exp_avg_sq"].float()
    c1 = 1 - 0.9 ** step
    c2 = 1 - 0.99 ** step
    update = lr * (m / c1) / (torch.sqrt(v / c2) + 1e-6)
    p_new = (p + update) / (1 - lr * weight_decay)
    p_new = torch.where(torch.isfinite(p_new), p_new, torch.zeros_like(p_new))
    m_new = (m - (1 - 0.9) * g) / 0.9
    v_new = (v - (1 - 0.99) * g * g) / 0.99
    rtol, atol = 1e-5, 1e-6
    for name, value in (("param", p_new), ("exp_avg", m_new), ("exp_avg_sq", v_new)):
        torch.testing.assert_close(inputs[name].float(), value, rtol=rtol, atol=atol,
                                   equal_nan=True)


def compute_ref_performance(**kwargs):
    return checks.measure(compute_ref, kwargs, torch.device("cpu"))


def compute_res_performance(**kwargs):
    return checks.measure(compute_res, kwargs, torch.device("cpu"))


@label("v1_cpu_adamw_rollback_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_cpu_adamw_rollback_(config):
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
