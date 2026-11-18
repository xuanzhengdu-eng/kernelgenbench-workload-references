"""Accuracy and performance for v1_sgl_transfer_embedding_ranges_.

SGLang embedding-range transfer contract:

* exactly one of ``src``/``dst`` is CUDA, both are 2-D, same dtype, same row
  width and contiguous;
* ``src_starts``, ``dst_starts`` and ``lengths`` have equal length; lengths are
  non-negative, zero-length ranges are skipped, and the complete plan is
  validated before any asynchronous copy, so an invalid later range leaves the
  destination untouched;
* the operation writes into ``dst`` and returns ``None``; ``src`` is read-only.

The Solution is the pinned SGLang native operator
``torch.ops.sgl_kernel.transfer_embedding_ranges_direct`` through
``sgl_kernel.kvcacheio.transfer_embedding_ranges_direct`` (source
``python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu``).  Host-to-device and
device-to-host runs exercise the real CUDA copy path.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize


# (direction, src rows, dst rows, width, dtype,
#  src starts, dst starts, lengths)
COMBOS = [
    ("h2d", 12, 13, 7, torch.float32, (1, 7, -9), (3, 9, -8), (3, 2, 0)),
    ("d2h", 9, 12, 5, torch.bfloat16, (0, 4), (4, 0), (2, 3)),
    ("h2d", 6, 8, 16, torch.uint8, (0, 5), (6, 0), (1, 1)),
    ("d2h", 5, 7, 4, torch.float16, (), (), ()),
]


def input_build(config):
    direction, src_rows, dst_rows, width, dtype, src_starts, dst_starts, lengths = config
    fill = 251 if dtype == torch.uint8 else -5
    values = (torch.arange(src_rows * width).reshape(src_rows, width) % 127).to(dtype)
    return {
        "src": values,
        "dst": torch.full((dst_rows, width), fill, dtype=dtype),
        "src_starts": list(src_starts),
        "dst_starts": list(dst_starts),
        "lengths": list(lengths),
    }


def place(direction, data):
    """Place the two sides on host/device as the direction requires."""
    src = data["src"].pin_memory() if direction == "h2d" else data["src"].cuda()
    dst = data["dst"].cuda() if direction == "h2d" else data["dst"].pin_memory()
    return {
        "src": src,
        "dst": dst,
        "src_starts": list(data["src_starts"]),
        "dst_starts": list(data["dst_starts"]),
        "lengths": list(data["lengths"]),
    }


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_sgl_transfer_embedding_ranges_(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_sgl_transfer_embedding_ranges_(**kwargs)


def oracle(config, data):
    direction, src_rows, dst_rows, width, dtype, src_starts, dst_starts, lengths = config
    fill = 251 if dtype == torch.uint8 else -5
    source = (torch.arange(src_rows * width).reshape(src_rows, width) % 127).to(dtype)
    expected = torch.full((dst_rows, width), fill, dtype=dtype)
    for src_start, dst_start, length in zip(src_starts, dst_starts, lengths):
        for row in range(length):
            expected[dst_start + row] = source[src_start + row]
    return expected


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def compare(ref_out, res_out, config, inputs, ref_inputs, res_inputs):
    assert ref_out is None and res_out is None
    synchronize()
    expected = oracle(config, inputs)
    torch.testing.assert_close(res_inputs["dst"].cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(ref_inputs["dst"].cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(res_inputs["src"].cpu(), inputs["src"], rtol=0, atol=0)
    torch.testing.assert_close(ref_inputs["src"].cpu(), inputs["src"], rtol=0, atol=0)


def bench(config, fn, repeats=10):
    data = input_build(config)
    placed = place(config[0], data)
    fn(**placed)
    synchronize()
    samples = []
    for _ in range(repeats):
        current = place(config[0], data)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(**current)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def compute_ref_performance(config):
    return bench(config, compute_ref)


def compute_res_performance(config):
    return bench(config, compute_res)


@label("v1_sgl_transfer_embedding_ranges_")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_sgl_transfer_embedding_ranges_(config):
    inputs = input_build(config)
    ref_inputs = place(config[0], inputs)
    res_inputs = place(config[0], inputs)
    before = {
        "src": ref_inputs["src"].clone(),
        "dst_ptr": res_inputs["dst"].data_ptr(),
        "dst": res_inputs["dst"].clone(),
    }

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    assert res_inputs["dst"].data_ptr() == before["dst_ptr"]
    compare(ref_out, res_out, config, inputs, ref_inputs, res_inputs)

    # A valid first range followed by an out-of-bounds range must be rejected
    # before any copy, for both implementations.
    for fn in (compute_ref, compute_res):
        broken = place(config[0], inputs)
        broken["src_starts"] = [0, broken["src"].shape[0]]
        broken["dst_starts"] = [0, 0]
        broken["lengths"] = [1, 1]
        snapshot = broken["dst"].clone()
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            fn(**broken)
        synchronize()
        torch.testing.assert_close(broken["dst"], snapshot, rtol=0, atol=0)

    from sandbox.utils.accuracy_utils import CustomBenchmarkResult

    ref_time = compute_ref_performance(config)
    res_time = compute_res_performance(config)
    return CustomBenchmarkResult(
        ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time if res_time else float("inf")
    )
