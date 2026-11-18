"""Accuracy and performance for v1_scaled_masked_softmax (Megatron forward)."""
import pytest
import torch
import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.bias_activation_test_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# x=(batch,heads,S,S); mask=(batch,1,S,S) bool where True masks an entry.
COMBOS = [((4, 4, 32), dtype, mode, scale)
          for dtype in (torch.float16, torch.bfloat16)
          for mode in ('plain', 'mask', 'causal')
          for scale in (0.5, 1.0)
          for _ in (0,)] + [((4, 4, 64), torch.bfloat16, 'mask', 0.5)]


def input_build(config):
    case, dtype, mode, scale = config
    b, h, s = case
    generator = torch.Generator(device="cpu").manual_seed(20261109)
    x = torch.randn((b, h, s, s), dtype=dtype, generator=generator)
    mask = None
    if mode == 'mask':
        mask = (torch.arange(s)[None, :] > torch.arange(s)[:, None])
        mask = mask.view(1, 1, s, s).expand(b, 1, s, s).contiguous()
    return {"x": x, "scale": scale, "mode": mode, "mask": mask}


def compute_ref(x, scale, mode, mask):
    return kernelgenbench.baseline.v1_scaled_masked_softmax(x, scale, mode, mask)


def compute_res(x, scale, mode, mask):
    return kernelgenbench.solution.v1_scaled_masked_softmax(x, scale, mode, mask)


def compare(ref_out, res_out, config):
    (b, h, s), dtype, mode, _ = config
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == (b, h, s, s)
    assert res_out.dtype == ref_out.dtype == dtype
    torch.testing.assert_close(res_out, ref_out, rtol=2e-2, atol=3e-3)
    # Each row is a probability distribution over the unmasked keys.
    sums = res_out.float().sum(-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=2e-2, atol=2e-2)


def compute_ref_performance(x, scale, mode, mask):
    return measure(compute_ref, {"x": x, "scale": scale, "mode": mode, "mask": mask})


def compute_res_performance(x, scale, mode, mask):
    return measure(compute_res, {"x": x, "scale": scale, "mode": mode, "mask": mask})


@label("v1_scaled_masked_softmax")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_scaled_masked_softmax(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    # Forward is pure: x and the mask are untouched.
    torch.testing.assert_close(res_inputs["x"].cpu(), inputs["x"], rtol=0, atol=0)
    assert not torch._C._is_alias_of(res_out, res_inputs["x"])

    ref_time = compute_ref_performance(**inputs)
    res_time = compute_res_performance(**inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
