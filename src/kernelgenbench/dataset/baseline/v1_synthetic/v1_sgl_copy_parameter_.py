"""SGLang parameter copy: shape-checked copy with downcast policy, returns None.

Official source (pinned commit):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/parameter.py#L51-L79

``copy_with_check`` asserts identical shapes, copies when the dtypes match,
otherwise ranks the dtypes (fp8=0, fp16/bf16=1, fp32=2, fp64=3).  An unknown
dtype raises ``ValueError('Unsupported copy between dtypes: ...')``.  Copying
into a lower rank is a downcast and raises ``ValueError('Downcasting not
allowed: ...')`` unless ``SGLANG_QUANT_ALLOW_DOWNCASTING`` is set; the test
passes that flag explicitly as ``allow_downcast``.  On success it returns None,
unlike vLLM's ``copy_weight_`` which returns the destination.
"""
import torch

_RANK = {
    torch.float8_e4m3fn: 0,
    torch.float8_e4m3fnuz: 0,
    torch.float8_e5m2: 0,
    torch.float8_e5m2fnuz: 0,
    torch.float8_e8m0fnu: 0,
    torch.float16: 1,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
}


def v1_sgl_copy_parameter_(target, loaded_weight, allow_downcast=False):
    assert target.shape == loaded_weight.shape
    if target.dtype == loaded_weight.dtype:
        target.copy_(loaded_weight)
        return None
    target_rank = _RANK.get(target.dtype)
    loaded_rank = _RANK.get(loaded_weight.dtype)
    if target_rank is None or loaded_rank is None:
        raise ValueError(
            f"Unsupported copy between dtypes: target.dtype={target.dtype}, "
            f"loaded_weight.dtype={loaded_weight.dtype}")
    if target_rank < loaded_rank and not allow_downcast:
        raise ValueError(
            f"Downcasting not allowed: target.dtype={target.dtype}, "
            f"loaded_weight.dtype={loaded_weight.dtype}")
    target.copy_(loaded_weight)
    return None
