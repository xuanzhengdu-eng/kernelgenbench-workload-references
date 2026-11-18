"""Accuracy and performance for v1_ds_permute_rows (DeepSpeed EP permutation).

Contract: 2-D CUDA rows ``[N, H]`` plus 1-D int32 indices; a valid index copies
the row, a negative or ``>= N`` index produces an exactly zero row. The result
is a fresh ``[len(indices), H]`` tensor in the row dtype; inputs are untouched.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import moe_rows222_231_checks as checks


# (permuted_indices, number of source rows, hidden, dtype); the last two entries
# exercise out-of-range padding and an empty index list / empty source.
COMBOS = [
    ((2, -1, 0, 7, 1), 3, 8, torch.float32),
    ((1, 0), 2, 3, torch.bfloat16),
    ((-1, 9), 0, 4, torch.float32),
    ((), 0, 8, torch.float32),
    ((4, 3, 2, 1, 0), 5, 16, torch.float16),
    ((-1, -2), 1, 1, torch.float32),
]


def input_build(config):
    indices, num_rows, hidden, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(222)
    rows = torch.arange(num_rows * hidden, dtype=torch.float32).reshape(num_rows, hidden)
    rows = (rows * 0.25).to(dtype)
    permuted_indices = torch.tensor(indices, dtype=torch.int32)
    return {"rows": rows, "permuted_indices": permuted_indices}


def compute_ref(rows, permuted_indices):
    return kernelgenbench.baseline.v1_ds_permute_rows(rows, permuted_indices)


def compute_res(rows, permuted_indices):
    return kernelgenbench.solution.v1_ds_permute_rows(rows, permuted_indices)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == ref_out.device
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def compute_ref_performance(rows, permuted_indices):
    return checks.measure(
        kernelgenbench.baseline.v1_ds_permute_rows,
        {"rows": rows, "permuted_indices": permuted_indices},
        device=kernelgenbench.device,
    )


def compute_res_performance(rows, permuted_indices):
    return checks.measure(
        kernelgenbench.solution.v1_ds_permute_rows,
        {"rows": rows, "permuted_indices": permuted_indices},
        device=kernelgenbench.device,
    )


@label("v1_ds_permute_rows")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_permute_rows(config):
    indices, num_rows, hidden, dtype = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    assert ref_inputs["rows"].shape == (num_rows, hidden)
    assert ref_inputs["permuted_indices"].dtype == torch.int32
    before = {name: value.cpu().clone() for name, value in res_inputs.items()}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # Direct contract assertions: valid slots gather, padding slots are zero.
    source = res_inputs["rows"]
    expected = torch.zeros((len(indices), hidden), dtype=dtype, device=source.device)
    for position, index in enumerate(indices):
        if 0 <= index < num_rows:
            expected[position] = source[index]
    assert torch.equal(res_out, expected), "candidate did not gather/zero correctly"
    if res_out.numel() and res_inputs["rows"].numel():
        assert ref_out.data_ptr() != res_inputs["rows"].data_ptr()
        assert res_out.data_ptr() != res_inputs["rows"].data_ptr()
    for name, value in before.items():
        assert torch.equal(res_inputs[name].cpu(), value), f"{name} was modified"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
