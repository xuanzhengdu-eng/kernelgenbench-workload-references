"""Accuracy and performance for v1_row_logsumexp_topk (SGLang).

Item: SGLang row_logsumexp_topk, finite-input branch (row 75).
Official source (pinned commit):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/logsumexp.py#L247-L298

The four outputs are (maximum, log_sum, values, indices). ``log_sum`` is
log(sum(exp(x - max))) -- the normaliser relative to the max, NOT the absolute
logsumexp. Top-k ties prefer the lower column index because the sort is stable.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows72_81_checks as checks

NAME = "v1_row_logsumexp_topk"

# (rows, columns, k, dtype, strided) -- columns 17000 crosses the kernel block.
COMBOS = [
    (2, 32, 1, torch.float32, False),
    (3, 67, 3, torch.float32, True),
    (2, 17000, 8, torch.float32, False),
    (0, 32, 3, torch.float32, False),
    (2, 32, 1, torch.float16, False),
    (3, 67, 3, torch.bfloat16, True),
    (2, 17000, 8, torch.bfloat16, False),
    (4, 8, 8, torch.float16, False),
]
TOLERANCE = {
    torch.float32: (1e-5, 1e-5),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (2e-2, 1e-2),
}


def input_build(config):
    """x=(rows, columns) in [-1,1]; first 3 columns set to 2.0 so top-k has
    value ties (the lower index must win). Strided uses every other column."""
    rows, cols, k, dtype, strided = config
    gen = torch.Generator().manual_seed(20261109)
    x = torch.rand((rows, cols * (2 if strided else 1)), dtype=dtype,
                   generator=gen) * 2 - 1
    if strided:
        x = x[:, ::2]
    if rows:
        x[:, :min(3, cols)] = 2.0
    return {"x": x, "k": k}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_row_logsumexp_topk(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_row_logsumexp_topk(**inputs)


def compare(ref_out, res_out, inputs):
    assert isinstance(ref_out, tuple) and len(ref_out) == 4
    assert isinstance(res_out, tuple) and len(res_out) == 4
    rows, cols, k, dtype, strided = inputs["config"]
    expected_shapes = [(rows,), (rows,), (rows, k), (rows, k)]
    expected_dtypes = [torch.float32, torch.float32, torch.float32, torch.int64]
    for index, (a, b) in enumerate(zip(ref_out, res_out)):
        assert isinstance(b, torch.Tensor)
        assert b.shape == a.shape == expected_shapes[index]
        assert b.dtype == a.dtype == expected_dtypes[index]
        if index == 3:
            assert torch.equal(a, b), "top-k indices differ (stable tie break)"
        else:
            rtol, atol = TOLERANCE[dtype]
            torch.testing.assert_close(b, a, rtol=rtol, atol=atol)
    if rows == 0:
        return
    # Tie break: the three 2.0 values must occupy the first three indices in
    # ascending order on both sides.
    if k >= 3:
        assert torch.equal(
            res_out[3][:, :3],
            torch.arange(3, dtype=torch.int64,
                         device=res_out[3].device).expand(rows, 3))
    # log_sum is relative to the maximum, so exp(log_sum) must equal the sum of
    # the shifted exponentials; it is not the absolute logsumexp.
    x_cuda = inputs["x"].to(ref_out[0].device).float()
    shifted = torch.exp(x_cuda - ref_out[0][:, None]).sum(dim=1)
    torch.testing.assert_close(ref_out[1], torch.log(shifted), rtol=1e-5, atol=1e-5)


def compute_ref_performance(**inputs):
    return checks.measure(kernelgenbench.baseline.v1_row_logsumexp_topk,
                          inputs, kernelgenbench.device)


def compute_res_performance(**inputs):
    return checks.measure(kernelgenbench.solution.v1_row_logsumexp_topk,
                          inputs, kernelgenbench.device)


@label(NAME)
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_row_logsumexp_topk(config):
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    rows, cols, k, dtype, strided = config
    assert ref_inputs["x"].shape == (rows, cols)
    assert ref_inputs["x"].dtype == dtype
    if strided:
        assert ref_inputs["x"].stride(1) == 2
    before = ref_inputs["x"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, {"config": config, "x": ref_inputs["x"]})

    assert torch.equal(before, ref_inputs["x"]), "reference modified its input"
    assert torch.equal(inputs["x"].cuda(), res_inputs["x"])

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return checks.timings(ref_time, res_time)
