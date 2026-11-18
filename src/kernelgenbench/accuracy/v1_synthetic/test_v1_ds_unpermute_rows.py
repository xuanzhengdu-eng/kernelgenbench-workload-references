# Synthetic workload; no production capture or production frequency claim.
"""OP-c4159a95fc8b: DeepSpeed unpermute_rows forward.

Inputs: `rows` is the padded, expert-major buffer [num_permuted_rows, hidden];
`permuted_indices` contains each of 0 .. num_rows - 1 exactly once plus -1 /
>= num_rows padding entries. The output restores the [num_rows, hidden] source
order; padding rows are never read. Key assertions are the exact restored
values, shape/dtype and the untouched inputs.
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
    return kernelgenbench.baseline.v1_ds_unpermute_rows(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_unpermute_rows(**inputs)


def compare(ref_out, res_out, dtype, num_rows, hidden, rows, indices):
    assert torch.is_tensor(res_out)
    assert res_out.shape == ref_out.shape == (num_rows, hidden)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == ref_out.device
    assert torch.equal(res_out, ref_out), "restored rows differ"
    # Independent oracle: source position of original row i is where the
    # complete permutation places i; padding entries never supply a row.
    if num_rows:
        positions = torch.stack([(indices == i).nonzero()[0, 0] for i in range(num_rows)])
        torch.testing.assert_close(ref_out, rows.index_select(0, positions), rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_ds_unpermute_rows")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_unpermute_rows(config):
    num_rows, hidden, dtype, padding = config
    inputs = input_build(config)
    ref_inputs = _cuda(inputs)
    res_inputs = _cuda(inputs)
    before = res_inputs["rows"].clone()
    indices_before = res_inputs["permuted_indices"].clone()
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype, num_rows, hidden,
            res_inputs["rows"], res_inputs["permuted_indices"])
    if res_out.numel() and res_inputs["rows"].numel():
        assert res_out.data_ptr() != res_inputs["rows"].data_ptr()
    assert torch.equal(res_inputs["rows"], before)
    assert torch.equal(res_inputs["permuted_indices"], indices_before)
    ref_time = compute_ref_performance(**ref_inputs)
    res_time = compute_res_performance(**res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
