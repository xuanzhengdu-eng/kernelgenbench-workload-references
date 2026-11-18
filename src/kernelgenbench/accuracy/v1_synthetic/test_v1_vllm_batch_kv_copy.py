"""Accuracy and timing for vLLM swap_blocks_batch (pointer/offset mode).

The solution binding converts the benchmark's (src buffer, src offset, dst
buffer, dst offset, nbytes) descriptors to CPU int64 pointer/size arrays and
calls vllm._custom_ops.swap_blocks_batch (torch.ops._C_cache_ops
.swap_blocks_batch), the real cuMemcpyBatchAsync/cudaMemcpyAsync path on H800.
The reference validates descriptor ranges and performs the same copies.  The
expected destination bytes are also rebuilt independently in the test.
"""
import statistics

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# Descriptors are (src buffer, src offset, dst buffer, dst offset, nbytes).
COMBOS = [
    (((0, 3, 0, 2, 7), (1, 0, 1, 5, 13), (0, 20, 2, 0, 8)), False),
    (((0, 1, 2, 3, 11), (0, 1, 1, 4, 11)), True),
    (((0, 0, 0, 0, 1), (1, 8, 1, 0, 16)), False),
    ((), False),
]


def input_build(config):
    descriptors, any_order = config
    src = [(torch.arange(32, dtype=torch.int32) + 37 * i).to(torch.uint8) for i in range(2)]
    dst = [torch.full((32,), 220 + i, dtype=torch.uint8) for i in range(3)]
    mapping = torch.tensor(descriptors, dtype=torch.int64).reshape(-1, 5)
    return {
        "src": src,
        "dst": dst,
        "block_mapping": mapping,
        "src_access_any": any_order,
    }


def _to(inputs):
    device = kernelgenbench.device
    return {
        "src": [tensor.to(device) for tensor in inputs["src"]],
        "dst": [tensor.to(device) for tensor in inputs["dst"]],
        "block_mapping": inputs["block_mapping"].clone(),
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_batch_kv_copy(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_batch_kv_copy(**inputs)


def compare(ref_out, res_out, ref_inputs, res_inputs, config):
    descriptors = config[0]
    assert ref_out is None and res_out is None
    expected = [tensor.clone() for tensor in ref_inputs["dst"]]
    for si, so, di, do, n in descriptors:
        expected[di][do:do + n] = ref_inputs["src"][si][so:so + n]
    for i in range(len(expected)):
        torch.testing.assert_close(ref_inputs["dst"][i], expected[i], rtol=0, atol=0)
        torch.testing.assert_close(res_inputs["dst"][i], expected[i], rtol=0, atol=0)
    for i in range(len(ref_inputs["src"])):
        torch.testing.assert_close(res_inputs["src"][i], ref_inputs["src"][i], rtol=0, atol=0)


def _bench(fn, inputs):
    samples = []
    for _ in range(6):
        args = _to(inputs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(**args)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples[1:])


def compute_ref_performance(**inputs):
    return _bench(compute_ref, inputs)


def compute_res_performance(**inputs):
    return _bench(compute_res, inputs)


@label("v1_vllm_batch_kv_copy")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_batch_kv_copy(config):
    inputs = input_build(config)
    ref_inputs = _to(inputs)
    res_inputs = _to(inputs)
    src_before = [tensor.clone() for tensor in res_inputs["src"]]
    assert compute_ref(**ref_inputs) is None
    assert compute_res(**res_inputs) is None
    compare(None, None, ref_inputs, res_inputs, config)
    for before, after in zip(src_before, res_inputs["src"]):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
