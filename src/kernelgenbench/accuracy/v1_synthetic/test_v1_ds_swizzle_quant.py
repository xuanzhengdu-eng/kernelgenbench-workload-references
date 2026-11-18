"""DeepSpeed swizzled quantization: grouped INT4/INT8, symmetry and swizzle order."""
import kernelgenbench
import pytest
import torch
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.rows422_431_utils import (
    assert_no_alias, assert_unchanged, fresh, measure)

# groups, bits, asymmetric, pipeline, nodes, devices.  Every group holds 512 FP16
# values; groups must be divisible by pipeline*nodes*devices.
COMBOS = [
    (8, 8, False, 1, 2, 4),
    (8, 4, False, 1, 2, 4),
    (8, 8, True, 1, 2, 4),
    (8, 4, True, 1, 2, 4),
    (16, 8, False, 2, 2, 4),
    (16, 4, True, 2, 2, 4),
    (8, 8, True, 2, 1, 2),
    (8, 4, False, 1, 1, 1),
    (2, 8, False, 1, 1, 2),
    (4, 4, True, 4, 1, 1),
]


def input_build(config):
    """Finite FP16 groups: an all-zero group, constant groups and a signed range."""
    groups, bits, asymmetric, pipeline, nodes, devices = config
    x = (((torch.arange(groups * 512, dtype=torch.float32) * 17) % 101) - 50) / 16
    x = x.reshape(groups, 512)
    x[0] = 0
    if groups > 1:
        x[1] = 0.75
    if groups > 2:
        x[2] = -0.5
    if groups > 3 and asymmetric:
        x[3] = 2.0
    return {
        "x": x.reshape(-1).to(torch.float16),
        "groups": groups,
        "bits": bits,
        "asymmetric": asymmetric,
        "pipeline": pipeline,
        "nodes": nodes,
        "devices": devices,
    }


def compute_ref(x, groups, bits, asymmetric, pipeline, nodes, devices):
    return kernelgenbench.baseline.v1_ds_swizzle_quant(
        x, groups, bits, asymmetric, pipeline, nodes, devices)


def compute_res(x, groups, bits, asymmetric, pipeline, nodes, devices):
    return kernelgenbench.solution.v1_ds_swizzle_quant(
        x, groups, bits, asymmetric, pipeline, nodes, devices)


def _swizzle(values, config):
    """Apply the pinned [pipeline, device, node, local_group] output order."""
    groups, bits, asymmetric, pipeline, nodes, devices = config
    local = groups // (pipeline * nodes * devices)
    return values.reshape(nodes, devices, pipeline, local, -1).permute(2, 1, 0, 3, 4).reshape(groups, -1)


def _unswizzle(values, config):
    """Inverse of the pinned [pipeline, device, node, local_group] output order."""
    groups, bits, asymmetric, pipeline, nodes, devices = config
    local = groups // (pipeline * nodes * devices)
    return values.reshape(pipeline, devices, nodes, local, -1).permute(2, 1, 0, 3, 4).reshape(groups, -1)


def _expected_scales(x, config):
    """Float32 statistics with the kernel's stored-scale formula, group-major."""
    groups, bits, asymmetric, pipeline, nodes, devices = config
    values = x.reshape(groups, 512).float()
    one = torch.ones((groups, 1), dtype=torch.float32)
    if asymmetric:
        hi = values.amax(-1, keepdim=True)
        lo = values.amin(-1, keepdim=True)
        inv = torch.where(hi == lo, one, (1 << bits) / (hi - lo))
        rows = torch.cat((1 / inv, (hi + lo) / 2), dim=1)
    else:
        maximum = values.abs().amax(-1, keepdim=True)
        inv = torch.where(maximum == 0, one, (1 << bits) / (2 * maximum))
        rows = 1 / inv
    return _swizzle(rows, config)


def _expected_bytes(x, config):
    """Explicit round/clamp/pack oracle for the int8 payload, before swizzling."""
    groups, bits, asymmetric, pipeline, nodes, devices = config
    values = x.reshape(groups, 512).float()
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    one = torch.ones((groups, 1), dtype=torch.float32)
    if asymmetric:
        hi = values.amax(-1, keepdim=True)
        lo = values.amin(-1, keepdim=True)
        offset = (hi + lo) / 2
        inv = torch.where(hi == lo, one, (1 << bits) / (hi - lo))
        q = torch.round((values - offset) * inv).clamp(qmin, qmax).to(torch.int16)
    else:
        maximum = values.abs().amax(-1, keepdim=True)
        inv = torch.where(maximum == 0, one, (1 << bits) / (2 * maximum))
        q = torch.round(values * inv).clamp(qmin, qmax).to(torch.int16)
    if bits == 4:
        packed = (((q[:, ::2] & 15) << 4) | (q[:, 1::2] & 15)).to(torch.uint8)
    else:
        packed = q.to(torch.int8).view(torch.uint8)
    return _swizzle(packed, config).reshape(-1)


def compare(ref_out, res_out, config):
    groups, bits, asymmetric, pipeline, nodes, devices = config
    for out in (ref_out, res_out):
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0].dtype == torch.int8
        assert out[0].shape == (groups * (512 if bits == 8 else 256),)
        assert out[1].dtype == torch.float32 and out[1].shape == (groups, 2 if asymmetric else 1)
    assert ref_out[0].device.type == res_out[0].device.type == kernelgenbench.device.type
    assert torch.equal(ref_out[0], res_out[0]), "packed bytes differ"
    assert torch.equal(ref_out[1], res_out[1]), "stored scales differ"


def compute_ref_performance(**inputs):
    return measure(compute_ref, inputs)


def compute_res_performance(**inputs):
    return measure(compute_res, inputs)


@label("v1_ds_swizzle_quant")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_swizzle_quant(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    saved = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    for side in (ref_inputs, res_inputs):
        assert_unchanged(side, saved)
    assert_no_alias(res_out, res_inputs)

    expected = _expected_scales(inputs["x"], config).to(kernelgenbench.device)
    assert torch.equal(res_out[1], expected), "stored scales differ from the pinned formula"
    assert torch.equal(res_out[0].view(torch.uint8), _expected_bytes(inputs["x"], config).to(res_out[0].device)), \
        "packed bytes differ from the round/clamp/pack formula"

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time, speedup=ref_time / res_time)
