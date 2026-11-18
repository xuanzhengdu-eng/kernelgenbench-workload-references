"""H800 accuracy/performance test for v1_vllm_scatter_states_.

Contract under test (vLLM f358f0578a83f4e1fe6fc49a492d3ae9362de8e5,
vllm/model_executor/layers/mamba/ops/scatter_states.py#L73-L106):
``state`` is a dense device tensor, ``src`` has the same trailing shape and one
row per index, and ``indices`` is a 1-D int32/int64 tensor.  The rows are
written into ``state`` in place (unique indices; the source kernel is
non-atomic); the call returns None.  ``src`` and ``indices`` are not modified.
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

# (row shape, state dtype, padded batch stride, index dtype)
COMBOS = [
    ((2, 4), torch.float32, False, torch.int32),
    ((2, 4), torch.float32, True, torch.int64),
    ((3, 8), torch.bfloat16, True, torch.int32),
    ((3, 8), torch.bfloat16, False, torch.int64),
    ((1, 16), torch.float16, True, torch.int32),
    ((1, 16), torch.float16, False, torch.int64),
    ((2, 3, 5), torch.float32, True, torch.int64),
]


def input_build(config):
    row_shape, dtype, padded, index_dtype = config
    gen = torch.Generator(device="cpu").manual_seed(538)
    state_base = torch.randn((3, 2) + row_shape, dtype=torch.float32, generator=gen).to(dtype)
    src_base = torch.randn((3, 2) + row_shape, dtype=torch.float32, generator=gen).to(dtype)
    if padded:
        state = state_base[:, 0]
        src = src_base[:, 0]
    else:
        state = state_base[:, 0].contiguous()
        src = src_base[:, 0].contiguous()
    return {
        "state": state,
        "src": src,
        "indices": torch.tensor([2, 0, 1], dtype=index_dtype),
    }


def compute_ref(state, src, indices):
    return kernelgenbench.baseline.v1_vllm_scatter_states_(state, src, indices)


def compute_res(state, src, indices):
    return kernelgenbench.solution.v1_vllm_scatter_states_(state, src, indices)


def compare(ref_out, res_out, ref_inputs, res_inputs, config):
    row_shape, dtype, padded, _ = config
    assert ref_out is None and res_out is None
    assert ref_inputs["state"].shape == res_inputs["state"].shape == (3,) + row_shape
    assert ref_inputs["state"].dtype == res_inputs["state"].dtype == dtype
    # In-place writes must be identical and must not replace the state storage.
    torch.testing.assert_close(
        res_inputs["state"], ref_inputs["state"], rtol=0, atol=0
    )
    # Source rows and the index vector are read-only inputs.
    torch.testing.assert_close(
        ref_inputs["src"], res_inputs["src"], rtol=0, atol=0
    )
    assert torch.equal(ref_inputs["indices"], res_inputs["indices"])


def compute_ref_performance(state, src, indices):
    return bench_cuda(lambda: compute_ref(state, src, indices), warmup=25, rep=100)


def compute_res_performance(state, src, indices):
    return bench_cuda(lambda: compute_res(state, src, indices), warmup=25, rep=100)


@label("v1_vllm_scatter_states_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_scatter_states_(config):
    row_shape, dtype, padded, _ = config
    inputs = input_build(config)
    assert inputs["state"].dtype == dtype
    if padded:
        assert inputs["state"].stride(0) > inputs["state"][0].numel()
    assert inputs["state"][0].is_contiguous()
    ref_inputs = clone_tree(inputs, kernelgenbench.device)
    res_inputs = clone_tree(inputs, kernelgenbench.device)
    ref_state_ptr = ref_inputs["state"].data_ptr()
    res_state_ptr = res_inputs["state"].data_ptr()
    ref_src = ref_inputs["src"].clone()
    res_src = res_inputs["src"].clone()
    ref_indices = ref_inputs["indices"].clone()
    res_indices = res_inputs["indices"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    assert ref_inputs["state"].data_ptr() == ref_state_ptr
    assert res_inputs["state"].data_ptr() == res_state_ptr
    compare(ref_out, res_out, ref_inputs, res_inputs, config)
    # The scatter must copy rows, not alias them into the source.
    assert torch.equal(ref_inputs["src"], ref_src)
    assert torch.equal(res_inputs["src"], res_src)
    assert torch.equal(ref_inputs["indices"], ref_indices)
    assert torch.equal(res_inputs["indices"], res_indices)
    assert not torch._C._is_alias_of(res_inputs["state"], res_inputs["src"])

    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time
    )
