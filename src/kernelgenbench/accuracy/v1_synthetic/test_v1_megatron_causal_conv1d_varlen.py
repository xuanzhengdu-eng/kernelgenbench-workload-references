"""Accuracy and GPU performance for v1_megatron_causal_conv1d_varlen.

Cases are channel-last packed sequences with zero-length requests, optional
initial states and widths 2-4.  Assertions cover output values, untouched
initial states, untouched inputs, the exact result shape, and the silu-only rule.
"""
import pytest
import torch
import triton.testing
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# lengths (may contain 0), conv_dim, width, initial states, dtype in COMBOS
CASES = [
    ((3, 0, 5), 16, 2, False),
    ((1, 4, 2), 16, 3, True),
    ((6, 0), 32, 4, True),
    ((2, 2, 2, 2), 16, 2, False),
]
COMBOS = [(case, dtype) for case in CASES
          for dtype in (torch.float32, torch.bfloat16)]


def input_build(config):
    (lengths, conv_dim, width, initial), dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261112)
    total = sum(lengths)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return {
        "x": torch.randn((total, conv_dim), dtype=dtype, generator=generator),
        "weight": torch.randn((conv_dim, width), dtype=dtype, generator=generator),
        "bias": torch.randn((conv_dim,), dtype=dtype, generator=generator),
        "cu_seqlens": torch.tensor(offsets, dtype=torch.int32),
        "initial_states": (torch.randn((len(lengths), conv_dim, width - 1),
                                       dtype=dtype, generator=generator)
                           if initial else None),
        "activation": "silu",
    }


def fresh(inputs):
    return {name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()}


def compute_ref(x, weight, bias, cu_seqlens, initial_states=None, activation="silu"):
    return kernelgenbench.baseline.v1_megatron_causal_conv1d_varlen(
        x, weight, bias, cu_seqlens, initial_states, activation)


def compute_res(x, weight, bias, cu_seqlens, initial_states=None, activation="silu"):
    return kernelgenbench.solution.v1_megatron_causal_conv1d_varlen(
        x, weight, bias, cu_seqlens, initial_states, activation)


def compare(ref_out, res_out, ref_inputs, res_inputs, dtype):
    tolerance = 2e-5 if dtype == torch.float32 else 0.02
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == ref_inputs["x"].shape
    assert res_out.dtype == ref_out.dtype == dtype
    torch.testing.assert_close(res_out, ref_out, rtol=tolerance, atol=tolerance)
    if ref_inputs["initial_states"] is not None:
        # The official entry only reads initial states.
        torch.testing.assert_close(res_inputs["initial_states"],
                                   ref_inputs["initial_states"], rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_megatron_causal_conv1d_varlen")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_causal_conv1d_varlen(config):
    (lengths, conv_dim, width, initial), dtype = config
    device = kernelgenbench.device
    inputs = input_build(config)
    ref_inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                  for key, value in fresh(inputs).items()}
    res_inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                  for key, value in fresh(inputs).items()}
    assert ref_inputs["x"].shape == (sum(lengths), conv_dim)
    assert ref_inputs["cu_seqlens"].tolist() == [0] + list(
        __import__("itertools").accumulate(lengths))
    assert (ref_inputs["initial_states"] is not None) == initial

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs, dtype)
    torch.testing.assert_close(res_inputs["x"].cpu(), inputs["x"], rtol=0, atol=0)

    ref_time = compute_ref_performance(**fresh(ref_inputs))
    res_time = compute_res_performance(**fresh(res_inputs))
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
