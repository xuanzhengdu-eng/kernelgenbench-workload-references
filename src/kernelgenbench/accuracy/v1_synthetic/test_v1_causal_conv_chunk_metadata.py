"""vLLM causal-conv chunk metadata: CPU scheduling values plus optional device arrays.

query_start_loc_p_cpu is int32/int64 on CPU and strictly nondecreasing.  nums is
one chunk count per batch entry (ceil(len/8)); mlist repeats the batch index and
offsetlist the within-batch chunk offset; both pointer arrays hold
2*max(1024,total) int32 entries padded with -1 on the requested device.
"""
import time
import pytest
import torch
import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

NAME = "v1_causal_conv_chunk_metadata"
CASES = [(0,), (0, 0, 1, 9, 26), (3, 11, 19, 20), (0, 8200)]
COMBOS = [(case, dtype, device) for case in CASES
          for dtype in (torch.int32, torch.int64)
          for device in ("cpu", "cuda")]


def input_build(config):
    case, dtype, device = config
    return {"query_start_loc_p_cpu": torch.tensor(case, dtype=dtype),
            "device": torch.device(device)}


def compute_ref(**kw):
    return kernelgenbench.baseline.v1_causal_conv_chunk_metadata(**kw)


def compute_res(**kw):
    return kernelgenbench.solution.v1_causal_conv_chunk_metadata(**kw)


def compare(ref_out, res_out, dtype):
    assert isinstance(res_out, tuple) and len(res_out) == 3
    ref_dict, ref_batch, ref_off = ref_out
    res_dict, res_batch, res_off = res_out
    assert list(ref_dict) == list(res_dict) == [8]
    for key in ("nums", "mlist", "offsetlist", "batch_ptr", "token_chunk_offset_ptr"):
        a, b = ref_dict[8][key], res_dict[8][key]
        assert isinstance(b, torch.Tensor) and a.device == b.device
        assert a.dtype == b.dtype and a.shape == b.shape, key
        assert torch.equal(a, b), key
    assert ref_dict[8]["nums"].dtype == dtype
    assert ref_dict[8]["tot"] == res_dict[8]["tot"]
    assert ref_dict[8]["mlist_len"] == res_dict[8]["mlist_len"]
    # The dict entries and the returned tuple must be the same objects.
    assert res_dict[8]["batch_ptr"] is res_batch
    assert res_dict[8]["token_chunk_offset_ptr"] is res_off
    assert torch.equal(ref_batch, res_batch) and torch.equal(ref_off, res_off)


def _bench(fn, data):
    for _ in range(3):
        fn(**data)
    if data["device"].type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        fn(**data)
    if data["device"].type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 100


def compute_ref_performance(data):
    return _bench(compute_ref, data)


def compute_res_performance(data):
    return _bench(compute_res, data)


@label(NAME)
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_causal_conv_chunk_metadata(config):
    case, dtype, device = config
    inputs = input_build(config)
    before = inputs["query_start_loc_p_cpu"].clone()
    ref_out = compute_ref(**inputs)
    res_out = compute_res(**inputs)
    compare(ref_out, res_out, dtype)
    torch.testing.assert_close(inputs["query_start_loc_p_cpu"], before, rtol=0, atol=0)
    ref_time = compute_ref_performance(inputs)
    res_time = compute_res_performance(inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
