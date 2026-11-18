"""H800 accuracy/performance test for v1_vllm_gather_initial_states.

Contract under test (vLLM f358f0578a83f4e1fe6fc49a492d3ae9362de8e5,
vllm/model_executor/layers/mamba/ops/gather_initial_states.py#L45-L83):
``state`` is a dense device tensor, ``indices`` is a 1-D int32/int64 tensor and
``has_initial_state`` is a 1-D bool tensor of the same length.  The result is a
new tensor with one row per index; rows whose flag is false are zero.  The
reference rows may be padded through ``state.stride(0)`` but every row itself
is contiguous.  Inputs are never mutated.
"""
import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows532_541_test_utils import (
    bench_cuda,
    clone_tree,
)

# (state row shape, dtype, padded batch stride, index dtype, flag mode)
COMBOS = [
    ((2, 4), torch.float32, False, torch.int32, "mixed"),
    ((2, 4), torch.float32, True, torch.int64, "mixed"),
    ((3, 8), torch.bfloat16, True, torch.int32, "mixed"),
    ((3, 8), torch.bfloat16, False, torch.int64, "all_false"),
    ((1, 16), torch.float16, True, torch.int32, "mixed"),
    ((1, 16), torch.float16, False, torch.int64, "all_false"),
    ((2, 3, 5), torch.float32, True, torch.int64, "mixed"),
]


def input_build(config):
    row_shape, dtype, padded, index_dtype, mode = config
    gen = torch.Generator(device="cpu").manual_seed(537)
    # A padded base gives state.stride(0) > rowsize while every row stays dense.
    base = torch.randn((3, 2) + row_shape, dtype=torch.float32, generator=gen).to(dtype)
    state = base[:, 0] if padded else base[:, 0].contiguous()
    indices = torch.tensor([2, 0, 1], dtype=index_dtype)
    if mode == "all_false":
        has_initial_state = torch.zeros(3, dtype=torch.bool)
    else:
        has_initial_state = torch.tensor([True, False, True])
    return {
        "state": state,
        "indices": indices,
        "has_initial_state": has_initial_state,
    }


def compute_ref(state, indices, has_initial_state):
    return kernelgenbench.baseline.v1_vllm_gather_initial_states(
        state, indices, has_initial_state
    )


def compute_res(state, indices, has_initial_state):
    return kernelgenbench.solution.v1_vllm_gather_initial_states(
        state, indices, has_initial_state
    )


def compare(ref_out, res_out, inputs, config):
    row_shape, dtype, padded, _, _ = config
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert ref_out.shape == res_out.shape == (inputs["indices"].numel(),) + row_shape
    assert ref_out.dtype == res_out.dtype == dtype
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)
    # The gather must allocate a result and leave its inputs untouched.
    for name, value in inputs.items():
        assert not torch._C._is_alias_of(res_out, value), f"output aliases {name}"
        assert not torch._C._is_alias_of(ref_out, value), f"reference aliases {name}"
    # Flags that are false are guaranteed to be exactly zero.
    false_rows = (~inputs["has_initial_state"]).nonzero().flatten().tolist()
    for row in false_rows:
        assert torch.count_nonzero(res_out[row]) == 0


def compute_ref_performance(state, indices, has_initial_state):
    return bench_cuda(
        lambda: compute_ref(state, indices, has_initial_state), warmup=25, rep=100
    )


def compute_res_performance(state, indices, has_initial_state):
    return bench_cuda(
        lambda: compute_res(state, indices, has_initial_state), warmup=25, rep=100
    )


@label("v1_vllm_gather_initial_states")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_gather_initial_states(config):
    row_shape, dtype, padded, _, _ = config
    inputs = input_build(config)
    assert inputs["state"].dtype == dtype
    if padded:
        assert inputs["state"].stride(0) > inputs["state"][0].numel()
    assert inputs["state"][0].is_contiguous()
    ref_inputs = clone_tree(inputs, kernelgenbench.device)
    res_inputs = clone_tree(inputs, kernelgenbench.device)
    before_ref = clone_tree(ref_inputs)
    before_res = clone_tree(res_inputs)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, ref_inputs, config)
    # Both implementations must leave every input (indices, flags, state) as-is.
    for name in inputs:
        assert torch.equal(ref_inputs[name], before_ref[name]), name
        assert torch.equal(res_inputs[name], before_res[name]), name

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
