"""Accuracy and performance for v1_shard_parameter (DeepSpeed aligned sharding).

Official implementation (pinned DeepSpeed 8af59b2e7a91c227081617240e98aa282eadc5f9):
  * ``deepspeed/inference/v2/model_implementations/sharding/utils.py#L13-L104``
    (``get_shard_endpoints`` + ``shard_param``)
  * ``deepspeed/inference/v2/model_implementations/sharding/embedding.py#L12-L21``
    (``shard_embedding_param`` -> inner-dimension ``shard_param``)

The inputs are the historical CASES: (mode, shape or None, number of
concatenated matrices, bias_dims, num_shards, shard_rank).  Every case is run
for float32/bfloat16 on CPU and CUDA.  The test asserts, per side:

  * ``None`` in -> ``None`` out and ``num_shards == 1`` in -> the *same object*
    out (both official early returns),
  * the exact aligned shard (values, shape, dtype) for every other case,
  * the sharded result is a fresh tensor, while the unsharded result aliases
    the input (object identity),
  * the input parameter is never modified.

``torch.chunk`` + ``narrow`` is restated here as an independent oracle for the
endpoints/remainder distribution, so a wrong axis or dropped remainder block is
caught even if both bindings agreed with each other.
"""
import statistics
import time

import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


GRANULARITY = 32

# (mode, shape or None, num_concatenated_matrices, bias_dims, num_shards, shard_rank)
CASES = [
    ("inner", (96, 128), 2, 1, 3, 0),
    ("inner", (96, 128), 2, 1, 3, 2),
    ("outer", (2, 128, 64), 2, 1, 3, 1),
    ("outer", (2, 128), 2, 2, 3, 2),
    ("inner", (64, 64), 1, 1, 1, 0),
    ("inner", (64, 96), 1, 1, 2, 0),
    ("inner", (64, 96), 1, 1, 2, 1),
    ("inner", None, 1, 1, 2, 0),
]

COMBOS = [
    (case, dtype, device)
    for case in CASES
    for dtype in (torch.float32, torch.bfloat16)
    for device in ("cpu", "cuda")
]


def input_build(config):
    """Fixed-seed CPU parameter plus the scalar configuration of this case."""
    (mode, shape, num_concatenated_matrices, bias_dims, num_shards, shard_rank), dtype, _ = config
    param = None
    if shape is not None:
        generator = torch.Generator(device="cpu").manual_seed(20261109)
        param = torch.randn(shape, dtype=dtype, device="cpu", generator=generator)
    return {
        "param": param,
        "mode": mode,
        "shard_rank": shard_rank,
        "num_shards": num_shards,
        "num_concatenated_matrices": num_concatenated_matrices,
        "granularity": GRANULARITY,
        "bias_dims": bias_dims,
    }


def compute_ref(**kwargs):
    return kernelgenbench.baseline.v1_shard_parameter(**kwargs)


def compute_res(**kwargs):
    return kernelgenbench.solution.v1_shard_parameter(**kwargs)


def _fresh(inputs, device):
    return {
        name: value.to(device=device, copy=True) if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def _expected(param, mode, num_concatenated_matrices, bias_dims, num_shards, shard_rank):
    """Independent restatement of the official endpoint/remainder contract."""
    if param is None or num_shards == 1:
        return param
    # inner shards the last dim; outer shards the last dim for bias (ndim ==
    # bias_dims) and the second-to-last dim for weight matrices.
    axis = -1 if mode == "inner" or param.ndim == bias_dims else -2
    chunk = param.shape[axis] // num_concatenated_matrices
    assert chunk % GRANULARITY == 0, "case must be granularity aligned"
    blocks = chunk // GRANULARITY
    base, remainder = divmod(blocks, num_shards)
    start = (shard_rank * base + min(shard_rank, remainder)) * GRANULARITY
    stop = start + (base + (1 if shard_rank < remainder else 0)) * GRANULARITY
    pieces = param.chunk(num_concatenated_matrices, dim=axis)
    return torch.cat([piece.narrow(axis, start, stop - start) for piece in pieces], dim=axis)


def compare(ref_out, res_out, ref_params, res_params, config, before):
    (mode, shape, num_concatenated_matrices, bias_dims, num_shards, shard_rank), dtype, device = config
    if shape is None:
        assert ref_out is None and res_out is None, "None parameter must shard to None"
    else:
        for name, output, params in (("reference", ref_out, ref_params), ("solution", res_out, res_params)):
            param = params["param"]
            assert isinstance(output, torch.Tensor), f"{name} must return a tensor"
            assert output.dtype == dtype, f"{name} dtype {output.dtype} != {dtype}"
            assert output.device == param.device, f"{name} device {output.device} != {param.device}"
            if num_shards == 1:
                # Official early return: the same object, not a copy.
                assert output is param, f"{name} must return the input object when num_shards == 1"
            else:
                assert not torch._C._is_alias_of(output, param), f"{name} shard must be a fresh tensor"
            expected = _expected(param, mode, num_concatenated_matrices, bias_dims, num_shards, shard_rank)
            assert output.shape == expected.shape, f"{name} shape {output.shape} != {expected.shape}"
            assert torch.equal(output, expected), f"{name} values differ from the aligned shard"
        assert ref_out.shape == res_out.shape and ref_out.dtype == res_out.dtype
        assert torch.equal(ref_out, res_out), "reference and solution shards differ"
        # Neither binding may modify the caller's parameter.
        assert torch.equal(ref_params["param"].cpu(), before), "reference modified its input"
        assert torch.equal(res_params["param"].cpu(), before), "solution modified its input"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(fn, inputs, device, repeats=5):
    """Median single-call wall time in milliseconds on ``device``."""
    fn(**_fresh(inputs, device))  # warmup
    _sync(device)
    samples = []
    for _ in range(repeats):
        arguments = _fresh(inputs, device)
        _sync(device)
        start = time.perf_counter()
        fn(**arguments)
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def compute_ref_performance(inputs, device):
    return _measure(compute_ref, inputs, device)


def compute_res_performance(inputs, device):
    return _measure(compute_res, inputs, device)


@label("v1_shard_parameter")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_shard_parameter(config):
    (mode, shape, num_concatenated_matrices, bias_dims, num_shards, shard_rank), dtype, device_name = config
    device = torch.device(device_name)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA combo needs a GPU"

    inputs = input_build(config)
    ref_params = _fresh(inputs, device)
    res_params = _fresh(inputs, device)
    before = None if inputs["param"] is None else inputs["param"].clone()

    if before is not None:
        assert ref_params["param"].shape == shape
        assert ref_params["param"].dtype == dtype
        assert res_params["param"].dtype == dtype
        assert torch.equal(ref_params["param"].cpu(), before)
        assert torch.equal(res_params["param"].cpu(), before)
        assert ref_params["param"] is not res_params["param"]

    ref_out = compute_ref(**ref_params)
    res_out = compute_res(**res_params)
    compare(ref_out, res_out, ref_params, res_params, config, before)

    ref_ms = compute_ref_performance(inputs, device)
    res_ms = compute_res_performance(inputs, device)
    return CustomBenchmarkResult(
        ref_time=ref_ms,
        res_time=res_ms,
        speedup=ref_ms / res_ms if res_ms else float("inf"),
    )
