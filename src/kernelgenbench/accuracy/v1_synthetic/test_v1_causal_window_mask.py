"""Accuracy and performance for v1_causal_window_mask (Megatron get_*_causal_mask).

Contract: sq, skv, window_size, device -> boolean (sq, skv) mask where True
means masked.  window_size=None selects the plain causal mask and requires
sq == skv; otherwise the bottom-right aligned band
[skv-sq-w0, skv-sq+w1] is kept and everything outside is masked.  The result is
allocated fresh on ``device`` and the scalar arguments are not modified.
"""
import kernelgenbench
import pytest
import torch

from kernelgenbench.accuracy.v1_synthetic import bias_activation_test_utils as checks
from sandbox.verifier.test_parametrize import label, parametrize


COMBOS = [
    (4, 4, None),      # plain causal
    (0, 0, None),      # empty causal
    (3, 7, (2, 0)),    # bottom-right aligned, asymmetric window
    (7, 3, (1, 2)),
    (5, 5, (0, 0)),    # diagonal-only window
    (4, 8, (3, 1)),
    (2, 6, (0, 0)),
]


def input_build(config):
    sq, skv, window = config
    return {"sq": sq, "skv": skv, "window_size": window, "device": "cuda"}


def compute_ref(sq, skv, window_size, device):
    return kernelgenbench.baseline.v1_causal_window_mask(sq, skv, window_size, device)


def compute_res(sq, skv, window_size, device):
    return kernelgenbench.solution.v1_causal_window_mask(sq, skv, window_size, device)


def expected_mask(sq, skv, window, device):
    i = torch.arange(sq, device=device)[:, None]
    j = torch.arange(skv, device=device)[None, :]
    if window is None:
        return j > i
    low = skv - sq - window[0]
    high = skv - sq + window[1]
    return (j - i < low) | (j - i > high)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, torch.Tensor)
    assert ref_out.shape == res_out.shape, "mask shape differs"
    assert ref_out.dtype == res_out.dtype == torch.bool, "mask dtype must be bool"
    assert torch.equal(ref_out, res_out), "mask values differ"


def compute_ref_performance(sq, skv, window_size, device):
    return checks.measure(compute_ref, {"sq": sq, "skv": skv,
                                        "window_size": window_size, "device": device})


def compute_res_performance(sq, skv, window_size, device):
    return checks.measure(compute_res, {"sq": sq, "skv": skv,
                                        "window_size": window_size, "device": device})


@label("v1_causal_window_mask")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_causal_window_mask(config):
    sq, skv, window = config
    inputs = input_build(config)
    ref_out = compute_ref(**inputs)
    res_out = compute_res(**inputs)
    compare(ref_out, res_out)

    assert res_out.shape == (sq, skv)
    assert res_out.device.type == "cuda"
    expected = expected_mask(sq, skv, window, res_out.device)
    assert torch.equal(res_out, expected), "causal/window mask differs"
    assert res_out.sum().item() == expected.sum().item()
    if window is None:
        # Strict upper triangle: the diagonal is visible.
        upper = torch.arange(sq, device=res_out.device)[:, None] < \
            torch.arange(skv, device=res_out.device)[None, :]
        assert not res_out[~upper].any(), "causal mask must keep the diagonal"
        assert res_out[upper].all(), "causal mask must hide the strict upper triangle"
    else:
        # Band is bottom-right aligned; outside the band is masked.
        band = ~res_out
        for row in range(sq):
            visible = band[row].nonzero().flatten().tolist()
            if visible:
                assert visible == list(range(min(visible), max(visible) + 1))

    return checks.timings(
        compute_ref_performance(**inputs),
        compute_res_performance(**inputs),
    )


@pytest.mark.parametrize("case", [(3, 5), (7, 2)])
def test_accuracy_v1_causal_window_mask_requires_square_causal(case):
    sq, skv = case
    with pytest.raises(AssertionError):
        compute_ref(sq, skv, None, "cuda")
    with pytest.raises(AssertionError):
        compute_res(sq, skv, None, "cuda")
