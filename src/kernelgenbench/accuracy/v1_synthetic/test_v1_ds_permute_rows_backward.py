# Synthetic workload; no production capture or production frequency claim.
"""OP-f60d05da0a08: DeepSpeed permute_rows backward (explicit gradient helper).

Inputs: `rows` is the incoming gradient of the padded, expert-major buffer
[num_permuted_rows, hidden]; `permuted_indices` contains each of
0 .. num_rows - 1 exactly once plus padding entries. `_PermuteRows.backward`
returns the inverse gather [num_rows, hidden]; the trailing two tuple entries
are None because the official autograd backward has no gradient for the index
inputs.
"""
import pytest
import torch
import triton.testing
import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from sandbox.utils.accuracy_utils import CustomBenchmarkResult

COMBOS = [
    (3, 8, torch.float32, 2),
    (2, 3, torch.bfloat16, 0),
    (0, 4, torch.float32, 2),
    (64, 128, torch.float16, 5),
]


def _indices(num_rows, padding):
    generator = torch.Generator(device="cpu").manual_seed(1000 + num_rows + padding)
    core = torch.randperm(num_rows, generator=generator).to(torch.int32)
    pad = torch.full((padding,), -1, dtype=torch.int32)
    return torch.cat([core, pad, torch.tensor([num_rows + 3], dtype=torch.int32)])


def input_build(config):
    num_rows, hidden, dtype, padding = config
    indices = _indices(num_rows, padding)
    rows = torch.arange(indices.numel() * hidden, dtype=torch.float32)
    rows = rows.reshape(indices.numel(), hidden).to(dtype)
    return {"rows": rows, "permuted_indices": indices, "num_rows": num_rows}


def _cuda(inputs):
    return {k: (v.to(kernelgenbench.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_permute_rows_backward(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_permute_rows_backward(**inputs)


def compare(ref_out, res_out, dtype, num_rows, hidden):
    assert isinstance(ref_out, tuple) and len(ref_out) == 3
    assert isinstance(res_out, tuple) and len(res_out) == 3
    assert ref_out[1] is None and ref_out[2] is None
    assert res_out[1] is None and res_out[2] is None
    assert res_out[0].shape == ref_out[0].shape == (num_rows, hidden)
    assert res_out[0].dtype == ref_out[0].dtype == dtype
    assert res_out[0].device == ref_out[0].device
    assert torch.equal(res_out[0], ref_out[0]), "inverse gather differs"


def compute_ref_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_ds_permute_rows_backward")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_permute_rows_backward(config):
    num_rows, hidden, dtype, padding = config
    inputs = input_build(config)
    ref_inputs = _cuda(inputs)
    res_inputs = _cuda(inputs)
    before = res_inputs["rows"].clone()
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype, num_rows, hidden)
    if res_out[0].numel() and res_inputs["rows"].numel():
        assert res_out[0].data_ptr() != res_inputs["rows"].data_ptr()
    assert torch.equal(res_inputs["rows"], before)
    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
