"""Accuracy and performance for v1_megatron_count_grad_zeros.

Megatron ``count_zeros_fp32`` on a single-rank NCCL group: missing, shared and
TP-replicated gradients are filtered, ``numel - count_nonzero`` is accumulated on
a CUDA int64 scalar and SUM-reduced over ``grad_stats_parallel_group``; the
result is a Python int.  The FSDP branch reads ``param.grad._local_tensor``
(covered with a duck-typed FSDP parameter).  Gradients and parameter metadata
must be untouched.

Multirank note: this assignment sees one H800, and two ranks on the same device
make NCCL abort with "Duplicate GPU detected", so the group is world size one.
The real NCCL all_reduce and the CUDA accumulator still execute.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows172_181_test_utils import measure_cuda, nccl_world
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (mode, use_decoupled_grad, dtype)
COMBOS = [
    ("ordinary", False, torch.float32),
    ("missing", False, torch.float32),
    ("shared", False, torch.float32),
    ("ordinary", True, torch.bfloat16),
    ("missing", True, torch.bfloat16),
    ("fsdp", False, torch.float32),
]


def input_build(config):
    mode, decoupled, dtype = config
    params = []
    for i in range(3):
        param = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
        param.tensor_model_parallel = (i == 0)
        param.shared = (i == 2 and mode == "shared")
        values = [[0, 1, 0, 4], [0, 0, 2, 3], [0, 0, 0, 1]][i]
        if not (mode == "missing" and i == 0):
            grad = torch.tensor(values, dtype=dtype)
            if decoupled:
                param.decoupled_grad = grad
            else:
                param.grad = grad
        params.append(param)
    if mode == "fsdp":
        # The official FSDP branch only needs __fsdp_param__ and grad._local_tensor.
        class _LocalGrad:
            def __init__(self, tensor):
                self._local_tensor = tensor

        class _FSDPParam:
            __fsdp_param__ = True

            def __init__(self, tensor):
                self.grad = _LocalGrad(tensor)

        params.append(_FSDPParam(torch.tensor([0, 0, 5, 6], dtype=dtype)))
    return {"parameters": params, "use_decoupled_grad": decoupled}


EXPECTED = {
    "ordinary": 2 + 2 + 3,
    "missing": 5,
    "shared": 2 + 2,
    "fsdp": 2 + 2 + 3 + 2,
}


def _to_cuda(config):
    mode, decoupled, dtype = config
    params = input_build(config)["parameters"]
    moved = []
    for i, param in enumerate(params):
        if mode == "fsdp" and i == 3:
            tensor = param.grad._local_tensor.to("cuda")
            wrapper = type("_LocalGrad", (), {})()
            wrapper._local_tensor = tensor
            fsdp = type("_FSDPParam", (), {})()
            fsdp.__fsdp_param__ = True
            fsdp.grad = wrapper
            moved.append(fsdp)
            continue
        new = torch.nn.Parameter(torch.zeros(param.shape, dtype=param.dtype, device="cuda"))
        new.tensor_model_parallel = param.tensor_model_parallel
        new.shared = param.shared
        grad = getattr(param, "decoupled_grad", None) if decoupled else param.grad
        if grad is not None:
            grad = grad.to("cuda")
            if decoupled:
                new.decoupled_grad = grad
            else:
                new.grad = grad
        moved.append(new)
    return moved


def compute_ref(parameters, use_decoupled_grad, group):
    return kernelgenbench.baseline.v1_megatron_count_grad_zeros(
        parameters, group, use_decoupled_grad, group)


def compute_res(parameters, use_decoupled_grad, group):
    return kernelgenbench.solution.v1_megatron_count_grad_zeros(
        parameters, group, use_decoupled_grad, group)


def compare(ref_out, res_out, mode):
    """Both sides must return the same Python int, equal to the case's zero count."""
    assert type(ref_out) is int and type(res_out) is int
    assert ref_out == res_out == EXPECTED[mode], (ref_out, res_out, mode)


def compute_ref_performance(parameters, use_decoupled_grad, group):
    """Time the reference entry on the real NCCL group."""
    return measure_cuda(lambda: compute_ref(parameters, use_decoupled_grad, group))


def compute_res_performance(parameters, use_decoupled_grad, group):
    """Time the solution entry on the same group and inputs."""
    return measure_cuda(lambda: compute_res(parameters, use_decoupled_grad, group))


@label("v1_megatron_count_grad_zeros")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_count_grad_zeros(config):
    mode, decoupled, dtype = config
    group = nccl_world()
    ref_params = _to_cuda(config)
    res_params = _to_cuda(config)

    def snapshot(params):
        out = []
        for param in params:
            grad = getattr(param, "decoupled_grad", None) if decoupled else getattr(param, "grad", None)
            if grad is not None and hasattr(grad, "_local_tensor"):
                grad = grad._local_tensor
            out.append(None if grad is None else grad.clone())
        return out

    ref_before, res_before = snapshot(ref_params), snapshot(res_params)
    ref_out = compute_ref(ref_params, decoupled, group)
    res_out = compute_res(res_params, decoupled, group)

    compare(ref_out, res_out, mode)
    for side, params, before in (("reference", ref_params, ref_before),
                                 ("solution", res_params, res_before)):
        for i, (param, saved) in enumerate(zip(params, before)):
            grad = getattr(param, "decoupled_grad", None) if decoupled else getattr(param, "grad", None)
            if grad is not None and hasattr(grad, "_local_tensor"):
                grad = grad._local_tensor
            if saved is None:
                assert grad is None, f"{side} created gradient {i}"
            else:
                assert grad.dtype == dtype
                torch.testing.assert_close(grad.cpu(), saved.cpu(), rtol=0, atol=0,
                                           msg=f"{side} modified gradient {i}")

    ref_time = compute_ref_performance(ref_params, decoupled, group)
    res_time = compute_res_performance(res_params, decoupled, group)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
