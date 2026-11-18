"""Accuracy and GPU performance for v1_pack_fp8_to_int32.

The 2-D FP8 tensor is repacked into int32 words.  ``size_k_first=True`` returns a
transposed copy; ``False`` returns a view that aliases the contiguous input.
Both the exact bytes and the storage-alias contract are asserted.
"""
import pytest
import torch
import triton.testing
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

CASES = [(64, 128), (128, 1024)]
COMBOS = [(shape, size_k_first) for shape in CASES for size_k_first in (False, True)]


def input_build(config):
    shape, _ = config
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    values = torch.randn(shape, dtype=torch.float32, generator=generator)
    return {"fp8_tensor": values.to(torch.float8_e4m3fn)}


def compute_ref(fp8_tensor, size_k_first):
    return kernelgenbench.baseline.v1_pack_fp8_to_int32(fp8_tensor, size_k_first)


def compute_res(fp8_tensor, size_k_first):
    return kernelgenbench.solution.v1_pack_fp8_to_int32(fp8_tensor, size_k_first)


def compare(ref_out, res_out, source, size_k_first):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.dtype == ref_out.dtype == torch.int32
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert torch.equal(res_out, ref_out)
    aliases = (res_out.untyped_storage().data_ptr()
               == source.untyped_storage().data_ptr())
    assert aliases == (not size_k_first), "storage alias contract changed"


def compute_ref_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_ref(**inputs), warmup=25, rep=100)


def compute_res_performance(**inputs):
    return triton.testing.do_bench(lambda: compute_res(**inputs), warmup=25, rep=100)


@label("v1_pack_fp8_to_int32")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_pack_fp8_to_int32(config):
    shape, size_k_first = config
    device = kernelgenbench.device
    inputs = input_build(config)
    source = inputs["fp8_tensor"].to(device)
    ref_inputs = {"fp8_tensor": source.clone(), "size_k_first": size_k_first}
    res_inputs = {"fp8_tensor": source.clone(), "size_k_first": size_k_first}
    assert ref_inputs["fp8_tensor"].shape == shape
    assert ref_inputs["fp8_tensor"].dtype == torch.float8_e4m3fn

    for invalid_input in (source.view(torch.uint8), source.unsqueeze(0)):
        for compute in (compute_ref, compute_res):
            with pytest.raises(AssertionError):
                compute(invalid_input, size_k_first)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, res_inputs["fp8_tensor"], size_k_first)

    assert torch.equal(res_inputs["fp8_tensor"].cpu().view(torch.uint8),
                       inputs["fp8_tensor"].view(torch.uint8))
    ref_time = compute_ref_performance(fp8_tensor=source.clone(), size_k_first=size_k_first)
    res_time = compute_res_performance(fp8_tensor=source.clone(), size_k_first=size_k_first)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
