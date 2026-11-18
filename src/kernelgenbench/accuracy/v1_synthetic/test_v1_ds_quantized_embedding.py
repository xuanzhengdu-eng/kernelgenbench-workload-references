"""Accuracy and performance for v1_ds_quantized_embedding (723-row 471).

Operator: DeepSpeed ``QuantizedEmbedding.forward`` (dequantize + ``F.embedding``).
Official sources (pinned commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/layers.py#L75-L108
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/inference/quantization/utils.py#L96-L165

Contract: INT4 packs two codes per uint8 (high nibble first) along the last
dimension; INT8 is one code per byte.  ``quant_scale``/``quant_min`` hold one
value per ``group_size`` block along ``group_dim`` and the decoded weight is
``code / scale + min``, cast to ``dtype``, then plain ``F.embedding`` with the
layer's ``padding_idx`` and the fixed norm/scale_grad/sparse arguments.

Backend dispatch (upstream code): the ``quantizer`` CUDA extension handles
``group_size % 8 == 0`` and group_dim == last axis and fp16 on the accelerator;
all other combinations take the generic torch path.  Both are covered below, and
the same shape is included with fp32 so the fallback branch is exercised for an
otherwise kernel-eligible layout.
"""
import pytest
import torch

import kernelgenbench
from kernelgenbench.accuracy.v1_synthetic.rows462_471_utils import fresh, measure
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
from sandbox.verifier.test_parametrize import label, parametrize


# (vocab, dim, bits, group_dim, group_size, dtype, token ids, padding idx)
# The first two are the fp16 kernel-eligible layouts; the fp32 twin of the first
# keeps the same shape and must take the generic path.  A zero-length id set is
# deliberately absent: torch.embedding rejects empty indices on this CUDA build
# with "invalid argument" for both sides, so it is not a usable contract sample.
# A group_size=16 fp16 layout is also absent on purpose: upstream's dispatch
# (group_size % 8 == 0) admits it, but the pinned int4 kernel raises
# cudaErrorInvalidValue for it (logs/debug-471b.log); group_size 16 is covered
# through the generic fp32 path instead.
COMBOS = [
    (16, 16, 4, 1, 8, torch.float16, (0, 15, 3, 3), 0),
    (16, 16, 8, 1, 8, torch.float16, (2, 5, 9), None),
    (16, 16, 4, 1, 8, torch.float32, (4, 5), None),
    (8, 32, 4, 1, 8, torch.float16, (1, 2, 3), None),
    (8, 32, 4, 1, 16, torch.float32, (1, 2, 3), None),
    (8, 8, 4, 1, 4, torch.float32, (0, 7, 2, 2), 0),
    (6, 8, 8, 1, 2, torch.float16, (5, 1, 1), None),
    (8, 6, 4, 0, 2, torch.bfloat16, (3, 0, 7), 0),
    (4, 4, 8, 0, 4, torch.float32, (2,), None),
    (12, 12, 8, 0, 6, torch.float32, (11, 0, 5), None),
    (8, 16, 4, 1, 8, torch.bfloat16, (7, 1), None),
]

TOLERANCE = {
    torch.float32: (1e-6, 1e-6),
    torch.float16: (2e-3, 1e-3),
    torch.bfloat16: (1e-2, 8e-3),
}


def _quantized(config):
    """Deterministic packed codes plus grouped scale/min (built on CPU)."""
    vocab, dim, bits, group_dim, group_size, dtype, ids, padding = config
    limit = 2 ** bits
    values = ((torch.arange(vocab * dim, dtype=torch.int32) * 3 + 1) % limit
              ).to(torch.uint8).reshape(vocab, dim)
    packed = ((values[:, ::2] << 4) | values[:, 1::2]) if bits == 4 else values
    shape = (vocab, dim // group_size, 1) if group_dim == 1 else \
        (vocab // group_size, 1, dim)
    groups = shape[0] * shape[1] * shape[2]
    scale = ((torch.arange(groups, dtype=torch.float32) % 5) + 2).to(dtype).reshape(shape)
    minimum = (((torch.arange(groups, dtype=torch.float32) % 3) - 1).to(dtype)
               .reshape(shape))
    return packed, scale, minimum


def input_build(config):
    vocab, dim, bits, group_dim, group_size, dtype, ids, padding = config
    packed, scale, minimum = _quantized(config)
    return {"indices": torch.tensor(ids, dtype=torch.int64), "packed": packed,
            "quant_scale": scale, "quant_min": minimum, "num_bits": bits,
            "group_dim": group_dim, "group_size": group_size, "dtype": dtype,
            "padding_idx": padding}


def compute_ref(**inputs):
    return kernelgenbench.baseline.v1_ds_quantized_embedding(**inputs)


def compute_res(**inputs):
    return kernelgenbench.solution.v1_ds_quantized_embedding(**inputs)


def oracle(config):
    """Independent fp64 statement: unpack codes, dequantize per group, gather."""
    vocab, dim, bits, group_dim, group_size, dtype, ids, padding = config
    packed, scale, minimum = _quantized(config)
    if bits == 4:
        codes = torch.empty((vocab, dim), dtype=torch.uint8)
        codes[:, 0::2] = packed >> 4
        codes[:, 1::2] = packed & 15
    else:
        codes = packed
    grouped = codes.view(*( (vocab, dim // group_size, group_size) if group_dim == 1
                           else (vocab // group_size, group_size, dim) ))
    decoded = (grouped.double() / scale.double() + minimum.double()).reshape(vocab, dim)
    index = torch.tensor(ids, dtype=torch.int64)
    return decoded[index]


def compare(ref_out, res_out, config):
    vocab, dim, bits, group_dim, group_size, dtype, ids, padding = config
    assert isinstance(res_out, torch.Tensor)
    assert res_out.shape == ref_out.shape == (len(ids), dim)
    assert res_out.dtype == ref_out.dtype == dtype
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out, ref_out, rtol=rtol, atol=atol)


@label("v1_ds_quantized_embedding")
@parametrize("config", COMBOS)
@pytest.mark.parametrize("config", COMBOS)
def test_accuracy_v1_ds_quantized_embedding(config):
    vocab, dim, bits, group_dim, group_size, dtype, ids, padding = config
    inputs = input_build(config)
    ref_inputs = fresh(inputs, kernelgenbench.device)
    res_inputs = fresh(inputs, kernelgenbench.device)

    ref_out = compute_ref(**ref_inputs)
    res_out = compute_res(**res_inputs)
    compare(ref_out, res_out, config)

    # Independent formulation in fp64, on the same inputs.
    expected = oracle(config).to(dtype).to(res_out.device)
    rtol, atol = TOLERANCE[dtype]
    torch.testing.assert_close(res_out, expected, rtol=rtol, atol=atol)

    # The packed weight is read-only for this entry and the output is new.
    torch.testing.assert_close(res_inputs["packed"].cpu(), inputs["packed"], rtol=0, atol=0)
    assert not torch._C._is_alias_of(res_out, res_inputs["packed"])

    ref_time = measure(compute_ref, ref_inputs)
    res_time = measure(compute_res, res_inputs)
    return CustomBenchmarkResult(ref_time=ref_time, res_time=res_time,
                                 speedup=ref_time / res_time)
