"""Accuracy and performance for v1_block_hadamard (723-row 99).

Reference: vLLM ``hadamard_matrix`` + ``block_hadamard_rotate``.
Official source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/input_hadamard.py#L24-L42

The candidate side calls those two official functions.  Blocks must be powers
of two that divide the last dimension; the transform is orthonormal, so the
block norm is preserved and the result is a fresh tensor of ``x.dtype``.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic import rows92_101_checks as checks


TOLERANCE = {
    torch.float32: (1e-5, 1e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (2e-2, 1e-2),
}

# (last dim, block): every block is a power of two dividing the last dim.
SHAPES = [((3, 32), 8), ((2, 4, 64), 16), ((3, 8), 8), ((5, 128), 128),
          ((2, 5, 16), 4), ((1, 512), 32)]

COMBOS = [
    (case, dtype)
    for case in SHAPES
    for dtype in (torch.float32, torch.float16, torch.bfloat16)
]


def input_build(config):
    (shape, block), dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261119)
    return {"x": torch.randn(shape, dtype=dtype, generator=generator),
            "block": block}


def compute_ref(x, block):
    return kernelgenbench.baseline.v1_block_hadamard(x, block)


def compute_res(x, block):
    return kernelgenbench.solution.v1_block_hadamard(x, block)


def compare(ref_out, res_out, dtype):
    checks.assert_tensor(res_out, ref_out, dtype, *TOLERANCE[dtype])


def compute_ref_performance(x, block):
    return checks.measure(kernelgenbench.baseline.v1_block_hadamard,
                          {"x": x, "block": block}, x.device)


def compute_res_performance(x, block):
    return checks.measure(kernelgenbench.solution.v1_block_hadamard,
                          {"x": x, "block": block}, x.device)


@label("v1_block_hadamard")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_block_hadamard(config):
    (shape, block), dtype = config
    inputs = input_build(config)
    ref_inputs = checks.fresh(inputs, kernelgenbench.device)
    res_inputs = checks.fresh(inputs, kernelgenbench.device)
    before = res_inputs["x"].clone()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)
    assert not torch._C._is_alias_of(res_out, res_inputs["x"])
    assert torch.equal(res_inputs["x"], before), "solution modified x"
    assert torch.equal(ref_inputs["x"], inputs["x"].to(ref_inputs["x"].device))

    # Orthonormal rotation: per-block energy is preserved (within the storage
    # rounding of the output dtype).
    scale = 1.0 if dtype == torch.float32 else 5e-2
    energy_in = res_inputs["x"].float().reshape(-1, block).pow(2).sum(-1)
    energy_out = res_out.float().reshape(-1, block).pow(2).sum(-1)
    torch.testing.assert_close(energy_out, energy_in, rtol=scale, atol=scale * block)

    # Timing uses the device tensors; the CPU dict above only holds sources.
    return checks.timings(compute_ref_performance(**ref_inputs),
                          compute_res_performance(**res_inputs))
