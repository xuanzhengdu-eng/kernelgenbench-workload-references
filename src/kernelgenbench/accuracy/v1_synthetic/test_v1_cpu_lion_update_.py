"""Accuracy and performance for v1_cpu_lion_update_ (row 160).

Official entry: DeepSpeed ``cpu_lion.lion_update`` at commit
8af59b2e7a91c227081617240e98aa282eadc5f9, csrc/lion/cpu_lion_impl.cpp
L21-L63 (scalar path) and L192-L219 (binding), using the AVX path when the
extension selects it.  The test checks in-place semantics, the single momentum
state, signed-zero direction, and the untouched gradient.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows_152_161_utils import (
    fresh,
    measure_stateful,
)
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (shape, step, lr, beta1, beta2, weight_decay, signed-zero pair)
CASES = [
    ((1,), 1, 0.1, 0.9, 0.99, 0.0, False),
    ((9,), 3, 0.03, 0.9, 0.99, 0.1, True),
    ((4, 7), 7, 0.01, 0.8, 0.95, 0.02, True),
    ((0,), 1, 0.1, 0.9, 0.99, 0.0, False),
]
COMBOS = [
    (case, dtype, state_fp32)
    for case in CASES
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
    for state_fp32 in (False, True)
]

TOLERANCE = {
    torch.float32: (1e-6, 1e-7),
    torch.float16: (5e-3, 2e-3),
    torch.bfloat16: (2e-2, 8e-3),
}


def input_build(config):
    """Build contiguous CPU param/grad/state with optional signed-zero pair."""
    case, dtype, state_fp32 = config
    shape, step, lr, beta1, beta2, weight_decay, zeros = case
    rng = torch.Generator(device="cpu").manual_seed(16000 + (hash(shape) % 997))
    param = torch.randn(shape, generator=rng).to(dtype)
    grad = torch.randn(shape, generator=rng).to(dtype)
    state = torch.randn(shape, generator=rng).to(torch.float32 if state_fp32 else dtype)
    if zeros and param.numel() >= 2:
        grad.view(-1)[0] = 0.0
        state.view(-1)[0] = 0.0
        grad.view(-1)[1] = -0.0
        state.view(-1)[1] = -0.0
    return {
        "param": param,
        "grad": grad,
        "state": state,
        "step": step,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": weight_decay,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_cpu_lion_update_(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_cpu_lion_update_(**inputs)


def compare(ref_out, res_out, ref_inputs, res_inputs, pointers, dtype):
    assert ref_out == res_out == 0
    rtol, atol = TOLERANCE[dtype]
    for name in ("param", "state"):
        assert ref_inputs[name].shape == res_inputs[name].shape
        assert ref_inputs[name].dtype == res_inputs[name].dtype
        torch.testing.assert_close(
            res_inputs[name].float(), ref_inputs[name].float(), rtol=rtol, atol=atol, msg=name
        )
    assert torch.equal(res_inputs["grad"], ref_inputs["grad"])
    assert torch.equal(torch.signbit(res_inputs["grad"]), torch.signbit(ref_inputs["grad"]))
    # Both sides update their own tensors in place: the storage pointer recorded
    # before the call must still be the one the tensor uses afterwards.
    for side, inputs in (("ref", ref_inputs), ("res", res_inputs)):
        for name in ("param", "state", "grad"):
            assert inputs[name].data_ptr() == pointers[side][name], f"{side}: {name} was replaced"


def compute_ref_performance(inputs):
    return measure_stateful(compute_ref, inputs, torch.device("cpu"))


def compute_res_performance(inputs):
    return measure_stateful(compute_res, inputs, torch.device("cpu"))


@label("v1_cpu_lion_update_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_cpu_lion_update_(config):
    case, dtype, state_fp32 = config
    shape = case[0]
    inputs = input_build(config)
    ref_inputs = fresh(inputs, torch.device("cpu"))
    res_inputs = fresh(inputs, torch.device("cpu"))
    for side in (ref_inputs, res_inputs):
        assert side["param"].shape == shape
        assert side["param"].dtype == dtype
        assert side["state"].dtype == (torch.float32 if state_fp32 else dtype)

    pointers = {
        side: {name: inputs[name].data_ptr() for name in ("param", "state", "grad")}
        for side, inputs in (("ref", ref_inputs), ("res", res_inputs))
    }
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs, pointers, dtype)

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
