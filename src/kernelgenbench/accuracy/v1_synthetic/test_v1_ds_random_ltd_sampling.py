"""Accuracy and performance for the GPT branch of v1_ds_random_ltd_sampling.

Pinned entry ``gpt_sample_tokens``: uniform sample of ``reserved_length``
positions per (layer, batch) row, ascending native sort, and a sliced attention
mask.  The seeded RNG stream must be consumed identically by both sides, so the
same seed is applied before each call and the resulting RNG state is compared.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.spec_decode_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (layers, batch size, sequence length, reserved length, attention mask present)
CASES = [
    (2, 2, 16, 8, True),
    (1, 3, 32, 7, False),
    (1, 1, 64, 64, True),
    (2, 4, 33, 1, True),
]
COMBOS = CASES
SEED = 145


def input_build(config):
    layers, batch_size, seq_length, reserved_length, with_mask = config
    mask = (torch.arange(batch_size * seq_length * seq_length, dtype=torch.float32)
            .reshape(batch_size, 1, seq_length, seq_length) if with_mask else None)
    return {
        "reserved_length": reserved_length,
        "seq_length": seq_length,
        "batch_size": batch_size,
        "layers": layers,
        "device": str(kernelgenbench.device),
        "attn_mask": mask,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_random_ltd_sampling(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_random_ltd_sampling(**inputs)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(ref_out, tuple) and isinstance(res_out, tuple)
    assert len(ref_out) == len(res_out) == 2
    assert isinstance(res_out[0], torch.Tensor) and res_out[0].dtype == torch.int32
    assert res_out[0].shape == ref_out[0].shape
    torch.testing.assert_close(res_out[0], ref_out[0], rtol=0, atol=0)
    if ref_out[1] is None:
        assert res_out[1] is None
    else:
        assert isinstance(res_out[1], torch.Tensor)
        assert res_out[1].shape == ref_out[1].shape and res_out[1].dtype == ref_out[1].dtype
        torch.testing.assert_close(res_out[1], ref_out[1], rtol=0, atol=0)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs, device=kernelgenbench.device, seed=SEED)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs, device=kernelgenbench.device, seed=SEED)


@label("v1_ds_random_ltd_sampling")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_random_ltd_sampling(config):
    layers, batch_size, seq_length, reserved_length, with_mask = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    if with_mask:
        assert ref_inputs["attn_mask"].shape == (batch_size, 1, seq_length, seq_length)
    else:
        assert ref_inputs["attn_mask"] is None

    torch.manual_seed(SEED)
    ref_out = compute_ref(**ref_inputs)
    ref_state = torch.cuda.get_rng_state()
    torch.manual_seed(SEED)
    res_out = compute_res(**res_inputs)
    res_state = torch.cuda.get_rng_state()
    compare(ref_out, res_out)

    sampled = res_out[0]
    assert sampled.dtype == torch.int32
    assert sampled.shape == (layers, batch_size, reserved_length)
    assert torch.equal(res_state, ref_state), "sampling consumed a different RNG stream"
    assert torch.all(sampled[..., 1:] >= sampled[..., :-1]), "positions must be sorted"
    assert int(sampled.min()) >= 0 and int(sampled.max()) < seq_length
    if with_mask:
        # The sliced view aliases the caller's mask, exactly as upstream.
        assert res_out[1].shape == (batch_size, 1, reserved_length, reserved_length)
        assert res_out[1].untyped_storage().data_ptr() == res_inputs["attn_mask"].untyped_storage().data_ptr()
    else:
        assert res_out[1] is None

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
