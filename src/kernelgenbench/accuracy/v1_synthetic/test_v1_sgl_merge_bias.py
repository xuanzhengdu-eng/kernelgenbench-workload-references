"""Accuracy and performance for v1_sgl_merge_bias.

Official operator: SGLang ``merge_bias_tensor`` (sampling/sampling_batch_info.py).
Each config states which side is present; the missing side must be filled with
``default`` for ``bs`` rows using the other side's trailing shape and dtype.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic import upstream_op_test_utils as util
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# bs1, bs2, trailing shape, presence bitmask (1=lhs, 2=rhs), dtype, default
COMBOS = [
    (2, 3, (5,), 3, torch.float32, 0.0),
    (1, 2, (3, 2), 1, torch.float16, -1.25),
    (3, 1, (4,), 2, torch.bfloat16, 0.5),
    (2, 4, (2,), 0, torch.float32, 1.0),
    (0, 2, (3,), 1, torch.int64, -2),
    (1, 0, (3,), 2, torch.float32, 3.0),
    (3, 3, (2, 2), 3, torch.bfloat16, 0.0),
]


def input_build(config):
    bs1, bs2, shape, presence, dtype, default = config
    generator = torch.Generator(device="cpu").manual_seed(20261115)

    def rows(count, offset):
        if dtype == torch.int64:
            values = torch.randint(0, 17, (count, *shape), generator=generator, dtype=dtype)
        else:
            values = torch.randn((count, *shape), generator=generator, dtype=dtype)
        return values + offset

    lhs = rows(bs1, 0) if presence & 1 else None
    rhs = rows(bs2, 5) if presence & 2 else None
    # ``device`` only places the filled side that is missing from the inputs.
    return {"lhs": lhs, "rhs": rhs, "bs1": bs1, "bs2": bs2,
            "device": str(kernelgenbench.device), "default": default}


def compute_ref(lhs, rhs, bs1, bs2, device, default):
    return kernelgenbench.baseline.v1_sgl_merge_bias(lhs, rhs, bs1, bs2, device, default)


def compute_res(lhs, rhs, bs1, bs2, device, default):
    return kernelgenbench.solution.v1_sgl_merge_bias(lhs, rhs, bs1, bs2, device, default)


def compare(ref_out, res_out, config):
    bs1, bs2, shape, presence, dtype, default = config
    if presence == 0:
        assert ref_out is None and res_out is None, "both sides missing must return None"
        return
    assert isinstance(res_out, torch.Tensor) and isinstance(ref_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == (bs1 + bs2, *shape)
    assert res_out.dtype == ref_out.dtype == dtype
    assert res_out.device == ref_out.device
    assert res_out.stride() == ref_out.stride()
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def compute_ref_performance(lhs, rhs, bs1, bs2, device, default):
    return util.measure(compute_ref, {"lhs": lhs, "rhs": rhs, "bs1": bs1, "bs2": bs2,
                                      "device": device, "default": default})


def compute_res_performance(lhs, rhs, bs1, bs2, device, default):
    return util.measure(compute_res, {"lhs": lhs, "rhs": rhs, "bs1": bs1, "bs2": bs2,
                                      "device": device, "default": default})


@label("v1_sgl_merge_bias")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_merge_bias(config):
    bs1, bs2, shape, presence, dtype, default = config
    inputs = input_build(config)
    ref_inputs = util.fresh(inputs, kernelgenbench.device)
    res_inputs = util.fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    # The operator only reads its inputs and returns a fresh concatenation.
    for side in (ref_inputs, res_inputs):
        for name in ("lhs", "rhs"):
            if inputs[name] is not None:
                torch.testing.assert_close(side[name].cpu(), inputs[name], rtol=0, atol=0)
    if presence and presence != 3:
        assert not torch._C._is_alias_of(res_out, res_inputs["lhs"] if presence & 1
                                         else res_inputs["rhs"]), "output aliases an input"
        filled = res_out[bs1:] if presence & 1 else res_out[:bs1]
        assert torch.all(filled == default), "missing side was not filled with default"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
