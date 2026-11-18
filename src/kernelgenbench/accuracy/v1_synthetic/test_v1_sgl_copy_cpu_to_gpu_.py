"""SGLang short int32 CPU-to-GPU in-place copy.

Reference: ``kernelgenbench.baseline.v1_sgl_copy_cpu_to_gpu_``.
Solution: pinned SGLang AOT kernel ``copy_to_gpu_no_ce``
(python/sglang/kernels/aot/csrc/elementwise/copy.cu#L21-L57), which copies a
fixed 32/64/72-element int32 host array into a device tensor with one kernel
launch.

Contract under test:
  * the call returns ``None`` and writes into the caller's device tensor;
  * the device values equal the host values, for pinned and pageable host
    buffers and for every supported length;
  * the host input and the device pointer are unchanged;
  * any other element count is rejected by both sides.
"""
import time

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# (element count, host buffer pinned)
COMBOS = [(count, pinned) for count in (32, 64, 72) for pinned in (False, True)]


def input_build(config):
    count, pinned = config
    values = torch.randint(
        -100000, 100000, (count,), dtype=torch.int32,
        generator=torch.Generator().manual_seed(20261110),
    )
    host = values.pin_memory() if pinned else values
    return {"input": host, "output": torch.full((count,), -7, dtype=torch.int32,
                                               device="cuda")}


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_sgl_copy_cpu_to_gpu_(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.triton.v1_sgl_copy_cpu_to_gpu_(**kwargs)


def _bench(fn, inputs):
    for _ in range(3):
        fn(**inputs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        fn(**inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / 20.0


@label("v1_sgl_copy_cpu_to_gpu_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_copy_cpu_to_gpu_(config):
    ref_inputs = input_build(config)
    res_inputs = input_build(config)
    pointer = res_inputs["output"].data_ptr()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    assert ref_out is None and res_out is None
    assert res_inputs["output"].data_ptr() == pointer
    expected = ref_inputs["input"].cuda()
    assert torch.equal(ref_inputs["output"], expected)
    assert torch.equal(res_inputs["output"], expected)
    assert torch.equal(res_inputs["input"], ref_inputs["input"])

    ref_time = _bench(compute_ref, ref_inputs)
    res_time = _bench(compute_res, res_inputs)
    return CustomBenchmarkResult(
        ref_time=ref_time,
        res_time=res_time,
        speedup=ref_time / res_time if res_time else float("inf"),
    )

def test_unsupported_copy_length():
    host = torch.zeros((33,), dtype=torch.int32)
    device = torch.zeros_like(host, device="cuda")
    with pytest.raises(AssertionError):
        compute_ref(input=host, output=device)
    with pytest.raises((AssertionError, RuntimeError)):
        compute_res(input=host, output=device)
