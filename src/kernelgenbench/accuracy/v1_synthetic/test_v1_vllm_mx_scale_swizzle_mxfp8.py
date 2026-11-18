"""Accuracy and performance for v1_vllm_mx_scale_swizzle_mxfp8.

Operator: vLLM ``swizzle_mxfp8_scale``.
Source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/mxfp8_utils.py#L16-L38

Contract: same F8_128x4 swizzle as the MXFP4 entry, expressed with the MXFP8
helper: ``[M, K//32]`` uint8 scales padded to 128 rows and 4 scale columns,
viewed ``[mt,4,32,kt,4]``, transposed to ``[mt,kt,32,4,4]`` and flattened.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.quant_rows392_401_checks import (
    fresh, measure, timings,
)


COMBOS = [(1, 32), (32, 128), (127, 96), (129, 160), (128, 256)]


def input_build(config):
    rows, k = config
    scales = ((torch.arange(rows * (k // 32), dtype=torch.int32) % 251) + 1).to(torch.uint8)
    return {"sf": scales.reshape(rows, k // 32), "M": rows, "K": k}


def compute_ref(sf, M, K):
    return kernelgenbench.baseline.v1_vllm_mx_scale_swizzle_mxfp8(sf, M, K)


def compute_res(sf, M, K):
    return kernelgenbench.solution.v1_vllm_mx_scale_swizzle_mxfp8(sf, M, K)


@label("v1_vllm_mx_scale_swizzle_mxfp8")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_vllm_mx_scale_swizzle_mxfp8(config):
    inputs = input_build(config)
    rows, k = config
    cols = k // 32
    mt, kt = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros((mt * 128, kt * 4), dtype=torch.uint8)
    padded[:rows, :cols] = inputs["sf"]
    expected = padded.reshape(mt, 4, 32, kt, 4).permute(0, 3, 2, 1, 4).contiguous().view(-1)

    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    for name, out in (("reference", ref_out), ("solution", res_out)):
        assert out.dtype == torch.uint8
        assert out.ndim == 1 and out.is_contiguous()
        assert out.numel() == mt * kt * 32 * 4 * 4
        assert torch.equal(out.cpu(), expected), f"{name} swizzle mapping differs"
    for name, side in (("reference", ref_inputs), ("solution", res_inputs)):
        assert torch.equal(side["sf"].cpu(), inputs["sf"]), f"{name} modified the input"

    return timings(measure(compute_ref, inputs, kernelgenbench.device),
                   measure(compute_res, inputs, kernelgenbench.device))
