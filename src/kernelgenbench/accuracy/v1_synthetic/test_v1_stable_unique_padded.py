"""Accuracy and performance for v1_stable_unique_padded (vLLM unique_static_unsorted).

Contract: x is an integer tensor with values in [0, M]; the selected axis is
``dim``.  Output has the same shape/dtype, keeps the first occurrence of each
non-``ignored_val`` value in order, left-packs them and pads the tail with
``pad_val`` (valid range [-1, M]).  x is not modified.
"""
import kernelgenbench
import pytest
import torch

from kernelgenbench.accuracy.v1_synthetic import bias_activation_test_utils as checks
from sandbox.verifier.test_parametrize import label, parametrize


M = 5
COMBOS = [
    ((3, 7), -1, 0, -1),
    ((3, 7), -1, 2, 5),
    ((2, 3, 8), 1, 0, -1),
    ((2, 3, 8), 0, 0, -1),
    ((5,), -1, 0, 5),
    ((4, 4), -1, 0, -1),   # every value ignored -> fully padded
]


def input_build(config):
    shape, dim, ignored_val, pad_val = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    # Values span [0, M]; 0 is the default skip sentinel.
    return {
        "x": torch.randint(0, M + 1, shape, dtype=torch.int64, generator=generator, device="cpu"),
        "M": M,
        "dim": dim,
        "ignored_val": ignored_val,
        "pad_val": pad_val,
    }


def compute_ref(x, M, dim, ignored_val, pad_val):
    return kernelgenbench.baseline.v1_stable_unique_padded(x, M, dim, ignored_val, pad_val)


def compute_res(x, M, dim, ignored_val, pad_val):
    return kernelgenbench.solution.v1_stable_unique_padded(x, M, dim, ignored_val, pad_val)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, torch.Tensor)
    assert ref_out.shape == res_out.shape and ref_out.dtype == res_out.dtype
    assert torch.equal(ref_out, res_out), "packed unique values/order differ"


def compute_ref_performance(x, M, dim, ignored_val, pad_val):
    return checks.measure(compute_ref, {"x": x, "M": M, "dim": dim,
                                        "ignored_val": ignored_val, "pad_val": pad_val})


def compute_res_performance(x, M, dim, ignored_val, pad_val):
    return checks.measure(compute_res, {"x": x, "M": M, "dim": dim,
                                        "ignored_val": ignored_val, "pad_val": pad_val})


@label("v1_stable_unique_padded")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_stable_unique_padded(config):
    shape, dim, ignored_val, pad_val = config
    dtype = torch.int64
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    assert ref_out.shape == shape and ref_out.dtype == dtype
    # Input is read-only.
    torch.testing.assert_close(res_inputs["x"].cpu(), inputs["x"], rtol=0, atol=0)
    # Exact first-occurrence packing, including pad-tail and ignored values.
    xp = inputs["x"].movedim(dim % inputs["x"].ndim, -1)
    rows = xp.reshape(-1, xp.shape[-1])
    for i, row in enumerate(rows.tolist()):
        ordered = []
        for value in row:
            if value != ignored_val and value not in ordered:
                ordered.append(value)
        out_row = res_out.movedim(dim % inputs["x"].ndim, -1).reshape(-1, xp.shape[-1])[i]
        assert out_row[:len(ordered)].tolist() == ordered
        if len(ordered) < out_row.numel():
            assert (out_row[len(ordered):] == pad_val).all()

    return checks.timings(
        compute_ref_performance(**inputs),
        compute_res_performance(**inputs),
    )


@pytest.mark.parametrize("pad_val", [M + 1, -2])
def test_accuracy_v1_stable_unique_padded_rejects_bad_pad(pad_val):
    """pad_val outside [-1, M] is rejected by both implementations."""
    inputs = {
        "x": torch.randint(0, M + 1, (3, 7), dtype=torch.int64,
                           generator=torch.Generator().manual_seed(20261109)),
        "M": M,
        "dim": -1,
        "ignored_val": 0,
        "pad_val": pad_val,
    }
    for compute in (compute_ref, compute_res):
        with pytest.raises(ValueError):
            compute(**{key: (value.to(kernelgenbench.device) if isinstance(value, torch.Tensor)
                             else value) for key, value in inputs.items()})
