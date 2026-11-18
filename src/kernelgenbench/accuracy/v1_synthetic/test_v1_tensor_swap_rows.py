"""Accuracy and performance for v1_tensor_swap_rows (Megatron tensor_swap).

Contract: x=(N,*tail), src_idxs/dst_idxs=(M,) int64 on the same device.  The
call returns None and swaps rows in place with simultaneous-assignment
semantics (both gathers happen before either scatter).  Overlapping index
lists therefore still exchange the two original rows instead of leaving a copy.
"""
import kernelgenbench
import pytest
import torch

from kernelgenbench.accuracy.v1_synthetic import bias_activation_test_utils as checks
from sandbox.verifier.test_parametrize import label, parametrize


COMBOS = [
    ((6, 8), torch.float32, (0, 2), (4, 5)),        # disjoint lists
    ((6, 8), torch.float32, (0, 2), (2, 4)),        # one row in both lists
    ((7,), torch.float32, (1, 3, 5), (5, 3, 1)),    # 1-D, reversed order
    ((6, 2, 4), torch.bfloat16, (0, 1), (1, 2)),    # 3-D tail
    ((0, 4), torch.float32, (), ()),                # empty selection
]


def input_build(config):
    shape, dtype, src, dst = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    return {
        "x": torch.randn(shape, dtype=dtype, generator=generator, device="cpu"),
        "src_idxs": torch.tensor(src, dtype=torch.int64),
        "dst_idxs": torch.tensor(dst, dtype=torch.int64),
    }


def compute_ref(x, src_idxs, dst_idxs):
    return kernelgenbench.baseline.v1_tensor_swap_rows(x, src_idxs, dst_idxs)


def compute_res(x, src_idxs, dst_idxs):
    return kernelgenbench.solution.v1_tensor_swap_rows(x, src_idxs, dst_idxs)


def compare(ref_out, res_out, dtype=None):
    assert ref_out is None and res_out is None, "tensor_swap must return None"


def expected_swap(before, src_idxs, dst_idxs):
    """Independent model of the two-phase swap used by the official entry."""
    expected = before.clone()
    if src_idxs.numel():
        src_rows = before[src_idxs].clone()
        dst_rows = before[dst_idxs].clone()
        expected[dst_idxs] = src_rows
        expected[src_idxs] = dst_rows
    return expected


def compute_ref_performance(x, src_idxs, dst_idxs):
    return checks.measure(compute_ref, {"x": x, "src_idxs": src_idxs, "dst_idxs": dst_idxs})


def compute_res_performance(x, src_idxs, dst_idxs):
    return checks.measure(compute_res, {"x": x, "src_idxs": src_idxs, "dst_idxs": dst_idxs})


@label("v1_tensor_swap_rows")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_tensor_swap_rows(config):
    shape, dtype, _, _ = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    # Both sides see the same swaps and the index tensors are untouched.
    assert torch.equal(ref_inputs["x"], res_inputs["x"]), "swap result differs"
    for key in ("src_idxs", "dst_idxs"):
        assert torch.equal(ref_inputs[key].cpu(), inputs[key]) and torch.equal(res_inputs[key].cpu(), inputs[key])
    assert ref_inputs["x"].shape == shape and ref_inputs["x"].dtype == dtype

    # The call mutates the caller's storage in place and returns None: model
    # the two-phase (gather both, then scatter both) swap independently.
    before = inputs["x"].to(kernelgenbench.device)
    expected = expected_swap(before, res_inputs["src_idxs"], res_inputs["dst_idxs"])
    torch.testing.assert_close(res_inputs["x"], expected, rtol=0, atol=0)
    # src/dst values are exchanged exactly (no float arithmetic involved).
    if res_inputs["src_idxs"].numel():
        torch.testing.assert_close(
            res_inputs["x"][res_inputs["src_idxs"]],
            before[res_inputs["dst_idxs"]], rtol=0, atol=0)

    return checks.timings(
        compute_ref_performance(**inputs),
        compute_res_performance(**inputs),
    )
