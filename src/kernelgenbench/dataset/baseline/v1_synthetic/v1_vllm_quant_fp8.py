"""vLLM QuantFP8.forward_native for dynamic tensor/token/group and static scales.

Source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/input_quant_fp8.py#L184-L267

The reference covers the ``forward_native`` contract only.  The cache
conversion entry (``vllm._custom_ops.convert_fp8``) has a different signature,
return value, and in-place destination, and is a separate item
(``v1_vllm_cache_convert_fp8``).  ``group_shape`` follows vLLM's GroupShape
tuples: ``(-1, -1)`` per tensor, ``(1, -1)`` per token, ``(1, g)`` per token
group of ``g`` columns, and a static scale may carry any other positive shape.
"""
import torch
import torch.nn.functional as F


def _group_broadcast(value, shape):
    for dim, target in enumerate(shape):
        current = value.shape[dim] if dim < value.ndim else 1
        if current != target and current != 1:
            if target % current:
                raise ValueError(f"scale extent {current} does not divide {target}")
            value = (
                value.unsqueeze(dim + 1)
                .expand(*value.shape[: dim + 1], target // current, *value.shape[dim + 1 :])
                .flatten(dim, dim + 1)
            )
    return value


def _prepare_static_scale(scale, x, group_shape):
    if scale.numel() == 1:
        return scale if tuple(group_shape) == (-1, -1) else scale.reshape(1, 1)
    if scale.ndim == 1:
        rows, cols = group_shape
        rows = rows if rows > 0 else x.shape[-2]
        cols = cols if cols > 0 else x.shape[-1]
        if rows == x.shape[-2]:
            return scale.unsqueeze(-2)
        if cols == x.shape[-1]:
            return scale.unsqueeze(-1)
        raise ValueError("1D scale cannot be aligned with x")
    return scale


def v1_vllm_quant_fp8(
    x,
    group_shape,
    scale=None,
    scale_ub=None,
    num_token_padding=None,
    use_ue8m0=False,
    column_major_scales=False,
):
    if x.ndim != 2:
        raise ValueError("2D input only")
    rows, cols = group_shape
    fpmax = torch.finfo(torch.float8_e4m3fn).max
    floor = 1.0 / (fpmax * 512.0)
    is_group = rows == 1 and cols >= 1
    is_per_token = rows == 1 and cols == -1
    is_per_tensor = rows == -1 and cols == -1

    if is_group and scale is None:
        width = x.shape[-1]
        groups = (width + cols - 1) // cols
        padded = groups * cols
        work = F.pad(x, (0, padded - width)) if padded != width else x
        blocks = work.reshape(-1, groups, cols)
        s = (blocks.abs().max(-1, keepdim=True)[0].float() / fpmax).clamp(min=floor)
        if use_ue8m0:
            s = torch.exp2(torch.ceil(torch.log2(s)))
        q = (
            (blocks / s)
            .clamp(-fpmax, fpmax)
            .to(torch.float8_e4m3fn)
            .reshape(x.shape[0], padded)[:, :width]
            .reshape(x.shape)
        )
        s = s.squeeze(-1).reshape(x.shape[0], groups)
        if column_major_scales:
            s = s.transpose(-2, -1).contiguous().transpose(-1, -2)
        return q, s

    if scale is None:
        if is_per_token:
            peak = x.abs().amax(-1).unsqueeze(-1).float()
            if scale_ub is not None:
                peak = peak.clamp(max=scale_ub)
        elif is_per_tensor:
            peak = x.abs().max().unsqueeze(-1).float()
        else:
            raise ValueError("invalid dynamic scale mode")
        resolved = (peak / fpmax).clamp(min=floor)
    else:
        if scale_ub is not None:
            raise ValueError("scale upper bound only for dynamic per-token scales")
        resolved = _prepare_static_scale(scale, x, group_shape)

    expanded = resolved.float()
    for dim, target in enumerate(x.shape):
        current = expanded.shape[dim] if dim < expanded.ndim else 1
        if current != 1 and current != target:
            if target % current:
                raise ValueError(f"scale extent {current} does not divide {target}")
            expanded = expanded.repeat_interleave(target // current, dim=dim)
    expanded = _group_broadcast(expanded, x.shape[-2:]) if expanded.ndim <= 2 else expanded
    q = (x.float() * expanded.reciprocal()).clamp(-fpmax, fpmax).to(torch.float8_e4m3fn)
    if num_token_padding is not None:
        q = F.pad(q, (0, 0, 0, max(num_token_padding - x.shape[0], 0)))
    return q, resolved
