"""Accuracy/performance for v1_tensor_byte_view (DeepSpeed cast_to_byte_tensor).

A tensor with at most one byte of payload is returned unchanged; otherwise the result is
a flat ``uint8`` view with ``nbytes`` elements aliasing the source data pointer.  The
source contents are unchanged, and the identity relation to the source is compared.
"""
import time

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

COMBOS = [
    ((0,), torch.float32),
    ((1,), torch.float32),
    ((1,), torch.uint8),
    ((1, 1), torch.float16),
    ((2, 8), torch.float32),
    ((2, 8), torch.bfloat16),
    ((3, 4), torch.int32),
    ((3, 4), torch.uint8),
]


def input_build(config):
    shape, dtype = config
    count = 1
    for size in shape:
        count *= size
    return {"src_tensor": torch.arange(count, dtype=torch.float32).to(dtype).reshape(shape)}


def prepare(inputs, device):
    return {"src_tensor": inputs["src_tensor"].to(device)}


def compute_ref(src_tensor):
    return kernelgenbench.baseline.v1_tensor_byte_view(src_tensor)


def compute_res(src_tensor):
    return kernelgenbench.solution.v1_tensor_byte_view(src_tensor)


def compare(ref_out, res_out, dtype=None):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.device == ref_out.device
    assert torch.equal(res_out, ref_out)


def measure(fn, src_tensor):
    """Pure metadata/view operation: synchronized host wall time, not a GPU kernel."""
    if kernelgenbench.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(200):
        fn(src_tensor=src_tensor)
    if kernelgenbench.device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / 200


def compute_ref_performance(src_tensor):
    """Per-call host time of the reference view construction."""
    return measure(compute_ref, src_tensor)


def compute_res_performance(src_tensor):
    """Per-call host time of the solution view construction."""
    return measure(compute_res, src_tensor)


@label("v1_tensor_byte_view")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_tensor_byte_view(config):
    shape, dtype = config
    inputs = input_build(config)
    ref_inputs = prepare(inputs, kernelgenbench.device)
    res_inputs = prepare(inputs, kernelgenbench.device)
    source = res_inputs["src_tensor"]
    before = source.clone()
    nbytes = source.numel() * source.element_size()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, dtype)

    # Value/data-pointer contract and the identity rule for <= 1 byte.
    assert res_out.data_ptr() == source.data_ptr()
    assert (ref_out is ref_inputs["src_tensor"]) == (res_out is source)
    if nbytes > 1:
        assert res_out.dtype == torch.uint8
        assert res_out.numel() == nbytes
        assert res_out.is_contiguous()
    else:
        assert res_out is source
    torch.testing.assert_close(source, before, rtol=0, atol=0)

    if config is COMBOS[0]:
        # Documented input domain: the reference requires a contiguous source.  Upstream
        # does not validate and would reinterpret the raw data_ptr (measured: a
        # (4,4)[:, ::2] view still yields 32 uint8 elements), so the guard is explicit.
        strided = torch.randn((4, 4), device=source.device, dtype=torch.float32)[:, ::2]
        try:
            compute_ref(strided)
        except AssertionError:
            pass
        else:
            raise AssertionError("reference must reject a non-contiguous source")

    ref_time = compute_ref_performance(source)
    res_time = compute_res_performance(source)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
