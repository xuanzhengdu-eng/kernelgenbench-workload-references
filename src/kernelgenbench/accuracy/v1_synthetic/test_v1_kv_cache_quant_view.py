"""vLLM quantized KV-cache view: FP4 packed bytes vs the plain FP8 view.

Reference: v1_kv_cache_quant_view. Non-FP4 inserts a size-1 dimension; FP4 builds
``as_strided`` over packed uint8 bytes (``head_dim//2`` data + ``head_dim//32``
scale bytes per token). Solution: the pinned
``vllm.model_executor.layers.sparse_attn_indexer.kv_cache_as_quant_view``.
Both must alias the caller's storage and reproduce the exact strides.
"""
import statistics

import kernelgenbench
import pytest
import torch

from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# cache shape (blocks, block_size, bytes_per_token), dtype, FP4 cache flag
COMBOS = [
    ((2, 16, 128), torch.uint8, False),
    ((2, 16, 128), torch.uint8, True),
    ((2, 128, 1024), torch.uint8, False),
    ((2, 128, 1024), torch.uint8, True),
    ((1, 1, 64), torch.uint8, True),
    ((3, 5, 256), torch.uint8, True),
    ((3, 5, 256), torch.uint8, False),
    ((1, 7, 96), torch.uint8, False),
]


def input_build(config):
    shape, dtype, use_fp4 = config
    blocks, block_size, head_dim = shape
    # FP4 needs head_dim//2 packed bytes plus head_dim//32 scale bytes per token;
    # keep an 8-byte guard so the as_strided request stays in bounds. The FP8
    # path stores exactly head_dim bytes.
    columns = head_dim + 8 if use_fp4 else head_dim
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    cache = torch.randint(0, 256, (blocks, block_size, columns), generator=generator, dtype=torch.uint8)
    return {"kv_cache": cache.to(dtype), "head_dim": head_dim, "use_fp4_cache": use_fp4}


def _fresh(inputs):
    device = kernelgenbench.device
    return {name: (value.clone().to(device) if isinstance(value, torch.Tensor) else value)
            for name, value in inputs.items()}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_kv_cache_quant_view(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_kv_cache_quant_view(**inputs)


def compare(ref_out, res_out):
    assert isinstance(ref_out, torch.Tensor) and isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype == torch.uint8
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0)


def _bench(function, inputs, repeats=5):
    fresh = _fresh(inputs)
    function(**fresh)
    samples = []
    for _ in range(repeats):
        fresh = _fresh(inputs)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(**fresh)
        end.record()
        end.synchronize()
        assert isinstance(result, torch.Tensor)
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)



def compute_ref_performance(**inputs):
    return _bench(compute_ref, inputs)


def compute_res_performance(**inputs):
    return _bench(compute_res, inputs)

@label("v1_kv_cache_quant_view")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_kv_cache_quant_view(config):
    shape, _, use_fp4 = config
    blocks, block_size, head_dim = shape
    inputs = input_build(config)
    ref_inputs = _fresh(inputs)
    res_inputs = _fresh(inputs)
    ref_ptr = ref_inputs["kv_cache"].untyped_storage().data_ptr()
    res_ptr = res_inputs["kv_cache"].untyped_storage().data_ptr()

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out)

    # The view aliases the candidate input's storage; no new payload tensor.
    assert res_out.untyped_storage().data_ptr() == res_ptr
    assert ref_out.untyped_storage().data_ptr() == ref_ptr
    assert res_ptr == res_inputs["kv_cache"].untyped_storage().data_ptr()
    expected_width = head_dim // 2 + head_dim // 32 if use_fp4 else head_dim
    assert res_out.shape == (blocks, block_size, 1, expected_width)
    assert res_inputs["kv_cache"].stride(0) == block_size * inputs["kv_cache"].shape[2]
    torch.testing.assert_close(res_inputs["kv_cache"], ref_inputs["kv_cache"], rtol=0, atol=0)

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)


if __name__ == "__main__":
    for combo in COMBOS:
        test_accuracy_v1_kv_cache_quant_view(combo)
    print("PASS", len(COMBOS), "kv_cache_quant_view combos")
