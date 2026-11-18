"""Accuracy and timing for vLLM connector per-layer KV block copy.

The solution binding calls the real pinned
vllm.distributed.kv_transfer.kv_connector.utils.copy_kv_blocks, which dispatches
to current_platform.insert_blocks_to_device (h2d) or
current_platform.swap_out_blocks_to_host (d2h).  Two layer widths, direction
h2d/d2h, and the upstream no-op cases (empty dict/list, mismatched lengths) are
covered.  Unique destination ids are used so assignment order is irrelevant.
"""
import statistics
import time

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (direction, kind) with kind in valid/empty/mismatch/src_empty/dst_empty
COMBOS = [
    ("h2d", "valid"),
    ("d2h", "valid"),
    ("h2d", "empty"),
    ("d2h", "mismatch"),
    ("h2d", "src_empty"),
    ("d2h", "dst_empty"),
]


def input_build(config):
    direction, kind = config
    src, dst = {}, {}
    for name, width in (("layer0", 8), ("layer1", 16)):
        src[name] = torch.arange(6 * width, dtype=torch.float32).reshape(6, width)
        dst[name] = torch.full((7, width), -17.0, dtype=torch.float32)
    if kind in ("valid", "src_empty", "dst_empty"):
        src_ids, dst_ids = [5, 0, 3], [1, 4, 6]
    elif kind == "empty":
        src_ids, dst_ids = [], []
    else:
        src_ids, dst_ids = [1, 2], [2]
    if kind == "src_empty":
        src = {}
    if kind == "dst_empty":
        dst = {}
    return {
        "src": src,
        "dst": dst,
        "src_block_ids": src_ids,
        "dst_block_ids": dst_ids,
        "direction": direction,
    }


def _to(inputs):
    device = kernelgenbench.device
    direction = inputs["direction"]
    src_device = torch.device("cpu") if direction == "h2d" else device
    dst_device = device if direction == "h2d" else torch.device("cpu")
    return {
        "src": {name: tensor.to(src_device) for name, tensor in inputs["src"].items()},
        "dst": {name: tensor.to(dst_device) for name, tensor in inputs["dst"].items()},
        "src_block_ids": list(inputs["src_block_ids"]),
        "dst_block_ids": list(inputs["dst_block_ids"]),
        "direction": direction,
    }


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_vllm_connector_kv_copy(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_vllm_connector_kv_copy(**inputs)


def compare(ref_out, res_out, ref_inputs, res_inputs, config):
    assert ref_out is None and res_out is None
    for name in ref_inputs["dst"]:
        expected = ref_inputs["dst"][name]
        actual = res_inputs["dst"][name]
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    for name in ref_inputs["src"]:
        torch.testing.assert_close(res_inputs["src"][name], ref_inputs["src"][name], rtol=0, atol=0)


def _bench(fn, inputs):
    samples = []
    for _ in range(6):
        args = _to(inputs)
        begin = time.perf_counter()
        fn(**args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - begin) * 1000.0)
    return statistics.median(samples[1:])


def compute_ref_performance(**inputs):
    return _bench(compute_ref, inputs)


def compute_res_performance(**inputs):
    return _bench(compute_res, inputs)


@label("v1_vllm_connector_kv_copy")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_connector_kv_copy(config):
    inputs = input_build(config)
    ref_inputs = _to(inputs)
    res_inputs = _to(inputs)
    src_before = {name: tensor.clone() for name, tensor in res_inputs["src"].items()}
    dst_before = {name: tensor.clone() for name, tensor in res_inputs["dst"].items()}
    assert compute_ref(**ref_inputs) is None
    assert compute_res(**res_inputs) is None
    compare(None, None, ref_inputs, res_inputs, config)
    for name, tensor in src_before.items():
        torch.testing.assert_close(res_inputs["src"][name], tensor, rtol=0, atol=0)
    if config[1] != "valid":
        for name, tensor in dst_before.items():
            torch.testing.assert_close(res_inputs["dst"][name], tensor, rtol=0, atol=0)
    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
