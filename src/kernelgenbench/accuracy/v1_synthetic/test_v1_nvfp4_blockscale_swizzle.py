"""Accuracy and GPU performance for the vLLM v1_nvfp4_blockscale_swizzle.

Inputs are logical (M,K) or (B,M,K) FP8 block-scale tensors.  Besides the padded
shape, dtype and exact byte layout, this test pins the vLLM-specific device rule:
the pinned body ends in ``.cuda()``, so even a CPU input must come back on the
current CUDA device.  The SGLang entry keeps the input device and is a separate
Item with its own test.
"""
import pytest
import torch
import triton.testing
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

CASES = [(1, 1), (7, 5), (128, 4), (129, 9), (2, 31, 7), (3, 128, 8), (0, 4), (4, 0)]
COMBOS = [(case,) for case in CASES]


def input_build(config):
    (shape,) = config
    count = 1
    for dim in shape:
        count *= dim
    values = torch.arange(count, dtype=torch.float32).remainder(32) / 8
    return {"scale": values.to(torch.float8_e4m3fn).reshape(shape)}


def compute_ref(scale):
    return kernelgenbench.baseline.v1_nvfp4_blockscale_swizzle(scale)


def compute_res(scale):
    return kernelgenbench.solution.v1_nvfp4_blockscale_swizzle(scale)


def expected_shape(shape):
    rows = (shape[-2] + 127) // 128 * 128
    cols = (shape[-1] + 3) // 4 * 4
    return ((shape[0], rows, cols) if len(shape) == 3 else (rows, cols))


def compare(ref_out, res_out, shape):
    assert res_out.shape == ref_out.shape == expected_shape(shape)
    assert res_out.dtype == ref_out.dtype == torch.float8_e4m3fn
    assert res_out.device == ref_out.device
    assert torch.equal(res_out.view(torch.uint8), ref_out.view(torch.uint8))


def compute_ref_performance(scale):
    return triton.testing.do_bench(lambda: compute_ref(scale), warmup=25, rep=100)


def compute_res_performance(scale):
    return triton.testing.do_bench(lambda: compute_res(scale), warmup=25, rep=100)


@label("v1_nvfp4_blockscale_swizzle")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_nvfp4_blockscale_swizzle(config):
    (shape,) = config
    inputs = input_build(config)
    assert inputs["scale"].device.type == "cpu"
    assert inputs["scale"].dtype == torch.float8_e4m3fn
    assert inputs["scale"].shape == shape

    device = kernelgenbench.device
    ref_inputs = {"scale": inputs["scale"].to(device)}
    res_inputs = {"scale": inputs["scale"].to(device)}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, shape)
    assert res_out.device.type == "cuda"

    # vLLM-specific device rule: a CPU input is promoted to the current CUDA
    # device by the pinned `.cuda()` call.  Both sides must obey it, so binding
    # the SGLang variant (which keeps CPU) as a candidate fails here.
    cpu_ref = compute_ref(scale=inputs["scale"])
    cpu_res = compute_res(scale=inputs["scale"])
    assert cpu_ref.device.type == "cuda", "pinned vLLM body must end in .cuda()"
    assert cpu_res.device.type == "cuda", "candidate must promote a CPU input to CUDA"
    assert torch.equal(cpu_ref.cpu().view(torch.uint8), cpu_res.cpu().view(torch.uint8))

    torch.testing.assert_close(res_inputs["scale"].cpu(), inputs["scale"], rtol=0, atol=0)

    ref_time = compute_ref_performance(inputs["scale"].to(device))
    res_time = compute_res_performance(inputs["scale"].to(device))
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
