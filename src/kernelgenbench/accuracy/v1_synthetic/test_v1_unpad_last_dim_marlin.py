"""Accuracy and performance for v1_unpad_last_dim_marlin.

Operator: vLLM ``marlin_unpad_output``.
Source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L288-L298

Contract: the caller states both extents.  With ``padded_n == size_n`` the input
object is returned unchanged; with a larger ``padded_n`` the N prefix is sliced
and made contiguous.  This is the same object/alias rule as the NVFP4 entry but
with a different signature (explicit padded extent), so it is a separate Item.
"""
import pytest
import torch

import kernelgenbench
from sandbox.verifier.test_parametrize import label, parametrize
from kernelgenbench.accuracy.v1_synthetic.quant_rows392_401_checks import (
    fresh, measure, timings,
)


# (rows, size_n, padded_n, dtype, layout)
COMBOS = [
    (128, 64, 128, torch.float32, "contiguous"),
    (128, 128, 128, torch.float32, "contiguous"),
    (128, 64, 64, torch.float16, "contiguous"),
    (16, 8, 16, torch.bfloat16, "strided"),
    (16, 16, 16, torch.float32, "strided"),
    (4, 0, 4, torch.float32, "contiguous"),
]


def input_build(config):
    rows, size_n, padded_n, dtype, layout = config
    generator = torch.Generator(device="cpu").manual_seed(717)
    if layout == "strided":
        wide = torch.randn((rows, padded_n * 2), dtype=dtype, generator=generator)
        out = wide[:, ::2]
    else:
        out = torch.randn((rows, padded_n), dtype=dtype, generator=generator)
    return {"output": out, "size_n": size_n, "padded_n": padded_n}


def compute_ref(output, size_n, padded_n):
    return kernelgenbench.baseline.v1_unpad_last_dim_marlin(output, size_n, padded_n)


def compute_res(output, size_n, padded_n):
    return kernelgenbench.solution.v1_unpad_last_dim_marlin(output, size_n, padded_n)


@label("v1_unpad_last_dim_marlin")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_unpad_last_dim_marlin(config):
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)
    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    for name, out, side in (("reference", ref_out, ref_inputs),
                            ("solution", res_out, res_inputs)):
        assert out.shape == (side["output"].shape[0], side["size_n"])
        assert out.dtype == side["output"].dtype
        if side["padded_n"] != side["size_n"]:
            assert out.is_contiguous()
        assert torch.equal(out.cpu(),
                           side["output"][..., :side["size_n"]].contiguous().cpu())
        if side["padded_n"] == side["size_n"]:
            assert out is side["output"], \
                f"{name} must return the input object when padded_n == size_n"
        else:
            assert out is not side["output"], f"{name} must return a new tensor object"
            slice_is_contiguous = side["output"][:, :side["size_n"]].is_contiguous()
            aliases = out.untyped_storage().data_ptr() == \
                side["output"].untyped_storage().data_ptr()
            assert aliases == slice_is_contiguous, \
                f"{name} storage sharing disagrees with prefix-slice contiguity"
        assert torch.equal(side["output"].cpu(), inputs["output"])

    return timings(measure(compute_ref, inputs, kernelgenbench.device),
                   measure(compute_res, inputs, kernelgenbench.device))
