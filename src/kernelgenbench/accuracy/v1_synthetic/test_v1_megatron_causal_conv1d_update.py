"""Accuracy and GPU performance for v1_megatron_causal_conv1d_update.

Each case builds channels-last x (batch, seq, dim) or (batch, dim), a per-slot conv_state of
state_len >= width - 1, and optional cache indices / intermediate buffer.  Besides
values it asserts the shifted cache, zero output for negative slots, the
untouched intermediate buffer when not supplied, and the string-only SiLU rule.
"""
import pytest
import torch
import triton.testing
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# batch, dim, seq, width, state_len, indexed, intermediate, bias, activation
CASES = [
    (2, 16, 1, 2, 2, False, False, True, "silu"),
    (2, 16, 3, 3, 3, False, True, True, "silu"),
    (3, 32, 2, 4, 5, True, False, False, None),
    (2, 16, 1, 3, 4, True, True, True, None),
    (2, 16, 2, 2, 3, False, True, True, True),
]
COMBOS = [(case, dtype) for case in CASES
          for dtype in (torch.float32, torch.bfloat16)]


def input_build(config):
    case, dtype = config
    batch, dim, seq, width, state_len, indexed, intermediate, bias, activation = case
    generator = torch.Generator(device="cpu").manual_seed(20261111)
    slots = batch + 1 if indexed else batch
    return {
        "x": torch.randn((batch, seq, dim), dtype=dtype, generator=generator),
        "conv_state": torch.randn((slots, dim, state_len), dtype=dtype, generator=generator),
        "weight": torch.randn((dim, width), dtype=dtype, generator=generator),
        "bias": (torch.randn((dim,), dtype=dtype, generator=generator)
                 if bias else None),
        "silu_activation": activation,
        "conv_state_indices": (torch.tensor([batch, -1] + [0] * (batch - 2),
                                            dtype=torch.int32)
                               if indexed else None),
        "intermediate_conv_states": (torch.full((slots, seq, dim, state_len), -5.0,
                                               dtype=dtype)
                                     if intermediate else None),
    }


def fresh(inputs):
    return {name: value.clone() if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()}


def compute_ref(x, conv_state, weight, bias=None, silu_activation=None,
                conv_state_indices=None, intermediate_conv_states=None):
    return kernelgenbench.baseline.v1_megatron_causal_conv1d_update(
        x, conv_state, weight, bias, silu_activation, conv_state_indices,
        intermediate_conv_states)


def compute_res(x, conv_state, weight, bias=None, silu_activation=None,
                conv_state_indices=None, intermediate_conv_states=None):
    return kernelgenbench.solution.v1_megatron_causal_conv1d_update(
        x, conv_state, weight, bias, silu_activation, conv_state_indices,
        intermediate_conv_states)


def compare(ref_out, res_out, ref_inputs, res_inputs, case, dtype):
    batch, dim, seq, _, _, indexed, intermediate, _, _ = case
    tolerance = 2e-5 if dtype == torch.float32 else 0.02
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == (batch, seq, dim)
    assert res_out.dtype == ref_out.dtype == dtype
    torch.testing.assert_close(res_out, ref_out, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(res_inputs["conv_state"], ref_inputs["conv_state"],
                               rtol=0, atol=0)
    if intermediate:
        torch.testing.assert_close(res_inputs["intermediate_conv_states"],
                                   ref_inputs["intermediate_conv_states"],
                                   rtol=0, atol=0)
        assert not torch.equal(res_inputs["intermediate_conv_states"],
                              torch.full_like(res_inputs["intermediate_conv_states"], -5.0))
    if indexed:
        # A negative slot writes zeros and must not touch its cached row.
        padded = (ref_inputs["conv_state_indices"] < 0).nonzero().flatten()
        for row in padded.tolist():
            assert torch.count_nonzero(res_out[row]) == 0


def compute_ref_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_megatron_causal_conv1d_update")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_megatron_causal_conv1d_update(config):
    case, dtype = config
    batch, dim, seq, width, state_len, indexed, intermediate, _, activation = case
    device = kernelgenbench.device
    inputs = input_build(config)
    ref_inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                  for key, value in fresh(inputs).items()}
    res_inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                  for key, value in fresh(inputs).items()}

    assert ref_inputs["x"].shape == (batch, seq, dim)
    assert ref_inputs["weight"].shape == (dim, width)
    assert ref_inputs["conv_state"].shape[-2:] == (dim, state_len)
    # state_len >= width keeps the shift loop non-degenerate; the pinned
    # Triton 3.6 compiler crashes on the state_len=1, width=2 form.
    assert state_len >= width
    assert (ref_inputs["conv_state_indices"] is not None) == indexed

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, res_inputs, case, dtype)

    # The 2-D single-token form is the same entry, not a second operator.
    ref_2d = compute_ref(ref_inputs["x"][:, 0].contiguous(),
                         ref_inputs["conv_state"].clone(), ref_inputs["weight"],
                         ref_inputs["bias"], activation,
                         ref_inputs["conv_state_indices"], None)
    assert ref_2d.shape == (batch, dim)
    if activation is True:
        # Boolean True must not activate: only the string "silu" does.
        plain = compute_ref(ref_inputs["x"], ref_inputs["conv_state"],
                            ref_inputs["weight"], ref_inputs["bias"], None,
                            ref_inputs["conv_state_indices"], None)
        torch.testing.assert_close(ref_2d, plain[:, 0], rtol=0, atol=0)

    ref_time = compute_ref_performance(**fresh(ref_inputs))
    res_time = compute_res_performance(**fresh(res_inputs))
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
