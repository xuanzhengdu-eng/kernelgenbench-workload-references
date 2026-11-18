"""SGLang Gemma3 scaled embedding: gather times a fixed float scale."""
import kernelgenbench
import pytest
import torch
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows422_431_utils import (
    assert_no_alias, assert_unchanged, fresh, measure)

# (index shape, vocabulary, embedding dim, scale, dtype)
COMBOS = [
    ((13,), 37, 64, 8.0, torch.float32),
    ((2, 11), 97, 128, 55.42562584220407, torch.float32),
    ((13,), 37, 64, 0.5, torch.float16),
    ((2, 11), 97, 128, 1.0, torch.bfloat16),
    ((1, 1), 5, 3, 3.0, torch.float32),
]


def input_build(config):
    shape, vocab, features, scale, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    table = torch.randn((vocab, features), dtype=dtype, generator=generator)
    indices = torch.randint(vocab, shape, generator=generator, dtype=torch.int64)
    # Force boundary and repeated indices without assuming more than one slot.
    flat = indices.reshape(-1)
    flat[0] = 0
    if flat.numel() > 1:
        flat[-1] = vocab - 1
    if flat.numel() > 2:
        flat[1] = 0
    return {"table": table, "indices": indices, "scale": scale}


def compute_ref(table, indices, scale):
    return kernelgenbench.baseline.v1_sglang_embedding_lookup(table, indices, scale)


def compute_res(table, indices, scale):
    return kernelgenbench.solution.v1_sglang_embedding_lookup(table, indices, scale)


def compare(ref_out, res_out, inputs):
    table, indices, scale = inputs["table"], inputs["indices"], inputs["scale"]
    assert res_out.dtype == ref_out.dtype == table.dtype
    assert res_out.shape == ref_out.shape == indices.shape + (table.shape[1],)
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)
    gathered = table.to(res_out.device)[indices.reshape(-1).to(res_out.device)] \
        .reshape(res_out.shape).float() * float(scale)
    torch.testing.assert_close(res_out.float(), gathered, rtol=1e-2, atol=1e-2)


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_sglang_embedding_lookup")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sglang_embedding_lookup(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    saved = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, inputs)

    # The scale must actually be applied: a scale of 0.5 halves the gathered row.
    if inputs["scale"] != 1.0:
        flat = inputs["indices"].reshape(-1)
        expected_row = inputs["table"].to(kernelgenbench.device)[flat.to(kernelgenbench.device)[0]].float() \
            * inputs["scale"]
        torch.testing.assert_close(res_out.reshape(-1, res_out.shape[-1])[0].float(), expected_row,
                                   rtol=1e-2, atol=1e-2)
    for side in (ref_inputs, res_inputs):
        assert_unchanged(side, saved)
    assert_no_alias(res_out, res_inputs)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
