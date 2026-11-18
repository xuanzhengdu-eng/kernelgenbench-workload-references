"""Mamba attention CP load balancing: reorder for attention and its inverse.

Non-packed cases cover the documented chunk orders for cp_size 1/2/3; packed cases
exercise the explicit plan adapter used when Transformer Engine is unavailable.
Both branches must return a new tensor with unchanged shape, dtype and device, and
must leave the caller's input and plan untouched.
"""
import pytest
import torch

import kernelgenbench
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize

# (sequence length, hidden width, cp size, packed, inverse)
COMBOS = [
    (seq, hidden, cp, False, inverse)
    for seq, hidden, cp in ((12, 8, 3), (16, 16, 2), (8, 8, 1), (24, 16, 4))
    for inverse in (False, True)
]
PACKED_COMBOS = [(12, 8, 3, True, inverse) for inverse in (False, True)]


def input_build(config, dtype):
    seq, hidden, cp, packed, inverse = config
    generator = torch.Generator(device="cpu").manual_seed(20261110)
    x = torch.randn((seq, 2, hidden), dtype=dtype, generator=generator)
    plan = torch.randperm(seq, generator=generator).reshape(cp, -1) if packed else None
    return {"input_": x, "cp_size": cp, "inverse": inverse, "packed_indices": plan}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_mamba_cp_rebalance(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_mamba_cp_rebalance(**inputs)


def _oracle(config, x, plan):
    seq, hidden, cp, packed, inverse = config
    if packed:
        idx = plan.reshape(-1).to(x.device)
        if not inverse:
            return x.index_select(0, idx)
        out = torch.empty_like(x)
        out[idx] = x
        return out
    chunks = torch.chunk(x, 2 * cp, dim=0)
    if inverse:
        order = [2 * i for i in range(cp)] + [2 * cp - 2 * i - 1 for i in range(cp)]
    else:
        order = [index for i in range(cp) for index in (i, 2 * cp - i - 1)]
    return torch.cat([chunks[index] for index in order], dim=0)


def compare(config, ref_out, res_out, inputs):
    seq, hidden, cp, packed, inverse = config
    expected = _oracle(config, inputs["input_"].to(ref_out.device),
                       None if inputs["packed_indices"] is None else inputs["packed_indices"].to(ref_out.device))
    for name, out in (("reference", ref_out), ("solution", res_out)):
        assert isinstance(out, torch.Tensor), name
        assert out.shape == expected.shape == (seq, 2, hidden)
        assert out.dtype == expected.dtype and out.device == expected.device
        assert torch.equal(out, expected), f"{name} differs from the documented order"
        assert out.data_ptr() != inputs["input_"].data_ptr(), f"{name} aliases the input"
    # The inverse really undoes the forward reorder.
    if not packed:
        forward = _oracle((seq, hidden, cp, packed, False), inputs["input_"].to(ref_out.device), None)
        back = _oracle((seq, hidden, cp, packed, True), forward, None)
        if inverse:
            assert torch.equal(back.cpu(), inputs["input_"])


@label("v1_mamba_cp_rebalance")
@parametrize("config", COMBOS + PACKED_COMBOS)
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("config,dtype", [(config, dtype)
                                          for config in COMBOS + PACKED_COMBOS
                                          for dtype in (torch.float32, torch.float16, torch.bfloat16)])
def test_accuracy_v1_mamba_cp_rebalance(config, dtype):
    inputs = input_build(config, dtype)
    device = torch.device("cuda")
    ref_inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                  for key, value in inputs.items()}
    res_inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                  for key, value in inputs.items()}
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(config, ref_out, res_out, inputs)
    assert torch.equal(ref_inputs["input_"].cpu(), inputs["input_"])
    assert torch.equal(res_inputs["input_"].cpu(), inputs["input_"])
    if inputs["packed_indices"] is not None:
        assert torch.equal(ref_inputs["packed_indices"].cpu(), inputs["packed_indices"])

    timed_ref = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in inputs.items()}
    timed_res = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in inputs.items()}

    def measure(call, kwargs):
        samples = []
        for _ in range(6):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call(**kwargs)
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        return sorted(samples)[len(samples) // 2]

    ref_time = measure(compute_ref, timed_ref)
    res_time = measure(compute_res, timed_res)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time if res_time else float("inf"))
