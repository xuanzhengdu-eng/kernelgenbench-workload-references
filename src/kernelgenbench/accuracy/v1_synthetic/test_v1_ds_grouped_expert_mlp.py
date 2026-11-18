"""Accuracy and performance for v1_ds_grouped_expert_mlp (DeepSpeed).

The official fixed commit exposes the sequential reference path
``_run_experts_for_loop``: split tokens by expert count, gate/up projections
with per-expert weights cast to the activation dtype, one of four registered
activations, down projection, then zero rows for the padding tail.  The test
covers zero-token experts, padding rows, FP32 weights on half activations and
all four activation forms.
"""
import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (counts, input width, hidden width, padding rows, activation, FP32 weights)
CASES = [
    ((2, 0, 3), 8, 12, 2, "swiglu", False),
    ((1, 4), 7, 9, 0, "geglu_tanh", True),
    ((0, 0), 4, 6, 3, "swiglu_clamped", False),
    ((2, 1), 5, 7, 1, "swiglu_oai", True),
    ((1, 1, 1), 8, 4, 0, "swiglu_clamped", False),
]
COMBOS = [
    (case, dtype)
    for case in CASES
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]


def input_build(config):
    case, dtype = config
    counts, width, hidden, padding, activation, fp32 = case
    rng = torch.Generator(device="cpu").manual_seed(20261110 + width)
    weight_dtype = torch.float32 if fp32 else dtype
    experts = len(counts)
    rows = sum(counts) + padding

    def rand(shape):
        return torch.randn(shape, dtype=dtype, generator=rng) * 0.5

    return {
        "x": rand((rows, width)),
        "w1": torch.randn((experts, hidden, width), dtype=weight_dtype, generator=rng) * 0.5,
        "w2": torch.randn((experts, width, hidden), dtype=weight_dtype, generator=rng) * 0.5,
        "w3": torch.randn((experts, hidden, width), dtype=weight_dtype, generator=rng) * 0.5,
        "num_tokens_per_expert": torch.tensor(counts, dtype=torch.int32),
        "activation": activation,
        "alpha": 1.702,
        "limit": 0.5,
    }


def _to_device(inputs):
    out = {}
    for key, value in inputs.items():
        out[key] = value.to(kernelgenbench.device) if isinstance(value, torch.Tensor) else value
    return out


def compute_ref(x, w1, w2, w3, num_tokens_per_expert, activation, alpha, limit):
    return kernelgenbench.baseline.v1_ds_grouped_expert_mlp(
        x, w1, w2, w3, num_tokens_per_expert, activation, alpha, limit
    )


def compute_res(x, w1, w2, w3, num_tokens_per_expert, activation, alpha, limit):
    return kernelgenbench.solution.v1_ds_grouped_expert_mlp(
        x, w1, w2, w3, num_tokens_per_expert, activation, alpha, limit
    )


def compare(ref_out, res_out, inputs, dtype, valid_rows):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == inputs["x"].shape
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == ref_out.device
    rtol, atol = {
        torch.float32: (2e-5, 2e-6),
        torch.float16: (2e-2, 1e-2),
        torch.bfloat16: (3e-2, 2e-2),
    }[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)
    if inputs["x"].shape[0] > valid_rows:
        assert bool((res_out[valid_rows:] == 0).all())
        assert bool((ref_out[valid_rows:] == 0).all())


def compute_ref_performance(x, w1, w2, w3, num_tokens_per_expert, activation, alpha, limit):
    return _bench(
        lambda: compute_ref(x, w1, w2, w3, num_tokens_per_expert, activation, alpha, limit)
    )


def compute_res_performance(x, w1, w2, w3, num_tokens_per_expert, activation, alpha, limit):
    return _bench(
        lambda: compute_res(x, w1, w2, w3, num_tokens_per_expert, activation, alpha, limit)
    )


def _bench(fn):
    import triton.testing

    return triton.testing.do_bench(fn, warmup=25, rep=100)


@label("v1_ds_grouped_expert_mlp")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_grouped_expert_mlp(config):
    case, dtype = config
    counts, width, hidden, padding, activation, fp32 = case
    inputs = input_build(config)
    experts = len(counts)
    valid_rows = sum(counts)
    assert inputs["x"].shape == (valid_rows + padding, width)
    assert inputs["w1"].shape == (experts, hidden, width)
    assert inputs["w2"].shape == (experts, width, hidden)
    assert inputs["w3"].shape == (experts, hidden, width)
    assert inputs["w1"].dtype == (torch.float32 if fp32 else dtype)
    before = {key: value.clone() for key, value in inputs.items()
              if isinstance(value, torch.Tensor)}

    ref_inputs = _to_device(inputs)
    res_inputs = _to_device(inputs)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, dtype, valid_rows)
    for side in (ref_inputs, res_inputs):
        for key, value in before.items():
            torch.testing.assert_close(side[key].cpu(), value, rtol=0, atol=0)

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
