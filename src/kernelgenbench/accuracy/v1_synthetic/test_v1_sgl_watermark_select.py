"""Accuracy and performance for v1_sgl_watermark_select.

Selection is deterministic: a Murmur3-style hash of the request key, the context
hash and the token id yields a log-uniform draw; the token maximising
``log(u) / probability`` wins.  Non-positive probabilities are ``-inf``, an
all-zero row therefore returns index 0, and the optional dual-key path mixes a
second key per request according to the mixing threshold.

Key assertions: int64 output of shape ``[batch]`` on the same device, exact token
equality (ties and float rounding are decided identically because both sides use
the same hash and score order), the all-zero row returns 0, and neither the
probabilities nor the hash inputs are modified.

Reference: Torch implementation of the fixed-commit watermark core.
Solution: SGLang's ``select_watermark_tokens_torch``.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.row_pair_utils import fresh, measure_cuda, to_device
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


CASES = [
    # batch, vocabulary, dual key, zero last row
    (3, 8, False, False),
    (3, 17, True, False),
    (3, 32, True, True),
    (1, 1, False, True),
]
COMBOS = [(batch, vocab, dual, zero, dtype)
          for (batch, vocab, dual, zero) in CASES
          for dtype in (torch.float32, torch.float16, torch.bfloat16)]


def input_build(config):
    batch, vocab, dual, zero, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    probabilities = torch.rand((batch, vocab), dtype=dtype, generator=generator)
    probabilities[:, ::5] = 0
    if zero:
        probabilities[-1].zero_()
    keys = torch.tensor([123, -1, 2 ** 40 + 1][:batch], dtype=torch.int64)
    context_hashes = torch.tensor([99, 2 ** 31 + 7, 1][:batch], dtype=torch.int64)
    keys_b = torch.tensor([456, 2 ** 40 + 3, -7][:batch], dtype=torch.int64) if dual else None
    thresholds = torch.tensor([0, 2 ** 31, 2 ** 32][:batch], dtype=torch.int64) if dual else None
    return {"probabilities": probabilities, "context_hashes": context_hashes,
            "keys": keys, "keys_b": keys_b, "mixing_thresholds": thresholds}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_sgl_watermark_select(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_sgl_watermark_select(**inputs)


def compare(ref_out, res_out, probabilities):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == (probabilities.shape[0],)
    assert res_out.dtype == ref_out.dtype == torch.int64
    assert res_out.device == ref_out.device
    assert torch.equal(res_out, ref_out), "selected tokens differ"
    assert int(res_out.min()) >= 0 and int(res_out.max()) < probabilities.shape[1]


def compute_ref_performance(inputs):
    return measure_cuda(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure_cuda(compute_res, inputs)


@label("v1_sgl_watermark_select")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_watermark_select(config):
    batch, vocab, dual, zero, dtype = config
    inputs = to_device(input_build(config))
    ref_inputs, res_inputs = fresh(inputs), fresh(inputs)
    assert ref_inputs["probabilities"].shape == (batch, vocab)
    assert (ref_inputs["keys_b"] is None) == (ref_inputs["mixing_thresholds"] is None)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, inputs["probabilities"])

    if zero:
        # All-masked row: every score is -inf and torch.argmax returns index 0.
        assert int(res_out[-1]) == 0
    for name in ("probabilities", "context_hashes", "keys"):
        torch.testing.assert_close(res_inputs[name], inputs[name], rtol=0, atol=0)

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))


@label("v1_sgl_watermark_select")
def test_invalid_dual_key_inputs():
    """Only one of key B / mixing thresholds is rejected by both sides."""
    base = to_device(input_build((3, 8, True, False, torch.float32)))
    for name in ("keys_b", "mixing_thresholds"):
        broken = dict(base)
        broken[name] = None
        with pytest.raises(ValueError):
            kernelgenbench.baseline.v1_sgl_watermark_select(**broken)
        with pytest.raises(ValueError):
            kernelgenbench.solution.v1_sgl_watermark_select(**broken)
