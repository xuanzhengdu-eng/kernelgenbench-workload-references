"""SGLang auxiliary hidden-state collection: list, passthrough and packer.

One Item, one reference/test.  The list strategy clones borrowed captures and
keeps an owned capture's identity; passthrough returns the tensor itself;
packer fills a preallocated buffer (or allocates one) and rejects a bad buffer.
"""
import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows712_723_utils import fresh, measure

# strategy, capture count, owned flags, preallocated buffer, tokens, width, dtype
COMBOS = [
    ("list", 1, (True,), False, 4, 8, torch.float32),
    ("list", 1, (False,), False, 3, 16, torch.bfloat16),
    ("list", 3, (True, False, True), False, 4, 8, torch.float32),
    ("packer", 3, (True, False, True), True, 5, 8, torch.float32),
    ("packer", 2, (False, False), False, 3, 16, torch.bfloat16),
    ("passthrough", 1, (True,), False, 3, 8, torch.bfloat16),
]


def input_build(config):
    strategy, count, owned, prealloc, rows, width, dtype = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    captures = [torch.randn((rows, width), dtype=dtype, generator=generator) for _ in range(count)]
    out = torch.zeros((rows, count * width), dtype=dtype) if prealloc else None
    return {"captures": captures, "owned": owned, "output_buffer": out, "strategy": strategy}


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_sgl_aux_capture_pack(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_sgl_aux_capture_pack(**kwargs)


def _aliases_any(output, tensors):
    return any(torch._C._is_alias_of(output, tensor) for tensor in tensors)


def compare(ref_out, res_out, ref_side, res_side, config):
    strategy, count, owned = config[0], config[1], config[2]
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)
    assert _aliases_any(res_out, res_side["captures"]) == _aliases_any(ref_out, ref_side["captures"])
    if strategy == "passthrough":
        assert res_out is res_side["captures"][0]
    elif strategy == "list":
        if count == 1 and owned[0]:
            assert res_out is res_side["captures"][0]
        else:
            assert not _aliases_any(res_out, res_side["captures"])
    else:
        if ref_side["output_buffer"] is not None:
            assert res_out is res_side["output_buffer"]
        else:
            assert not _aliases_any(res_out, res_side["captures"])
    for i in range(count):
        assert torch.equal(ref_side["captures"][i], res_side["captures"][i])


def compute_ref_performance(inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(inputs):
    return measure(compute_res, inputs)


@label("v1_sgl_aux_capture_pack")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_aux_capture_pack(config):
    inputs = input_build(config)
    ref_side, res_side = fresh(inputs), fresh(inputs)
    ref_before = [tensor.clone() for tensor in ref_side["captures"]]
    ref_out = compute_ref(**ref_side)
    res_out = compute_res(**res_side)
    compare(ref_out, res_out, ref_side, res_side, config)
    for before, actual in zip(ref_before, ref_side["captures"]):
        assert torch.equal(actual, before), "capture was modified"

    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
