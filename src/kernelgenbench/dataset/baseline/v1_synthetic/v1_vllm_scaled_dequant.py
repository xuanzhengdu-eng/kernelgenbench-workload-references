# v1_vllm_scaled_dequant: vLLM scaled_dequantize (native); 723-row 411.
"""Native scaled dequantization of an FP8 matrix with block or broadcast scales.

Source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/quant_utils.py#L507-L518

Contract
--------
``x_q`` is a ``float8_e4m3fn`` tensor; the shipped vLLM call sites pass a 2-D
matrix, and a 1-D vector is not rejected by the official entry either (the
scale broadcast makes it come back with a leading extent of 1, e.g. ``(8,)``
becomes ``(1, 8)``). ``group_shape`` is a
``GroupShape(row, col)`` (or ``None``); ``-1`` means "the full extent" of that
axis. The scale ``x_s`` may be:

* one element: used as a per-tensor scalar;
* 1-D: it must be accompanied by a group shape whose row extent equals
  ``x_q.shape[-2]`` (scale per row) or whose column extent equals
  ``x_q.shape[-1]`` (scale per column); otherwise ``ValueError``;
* 2-D: its extents are checked against ``x_q`` divided by the group shape.

After the shape checks the fp32 scale is broadcast with vLLM's "repeat each
element by ``target // n``" semantics (a dim of extent 1 broadcasts, a
non-matching extent must divide the target) and the product is cast to
``out_dtype`` (default float32). ``x_q`` and ``x_s`` are not modified, and a
new tensor is returned.

The cache-oriented ``convert_fp8`` branch of the old reference is a different
logical entry (writes into a caller-provided destination, takes a string cache
dtype, ignores ``group_shape``); it is a separate item,
``v1_vllm_cache_dequant_fp8``.
"""
import torch


def _normalize_group_shape(x, group_shape):
    return (group_shape[0] if group_shape[0] > 0 else x.shape[-2],
            group_shape[1] if group_shape[1] > 0 else x.shape[-1])


def _prep_scale(scale, x, group_shape):
    if scale.numel() == 1:
        per_tensor = group_shape is not None and group_shape[0] == -1 and group_shape[1] == -1
        return scale if per_tensor else scale.reshape(1, 1)
    if scale.ndim == 1:
        assert group_shape is not None, \
            "group_shape must be provided to correctly broadcast 1D scale"
        rows, cols = _normalize_group_shape(x, group_shape)
        if rows == x.shape[-2]:
            return scale.unsqueeze(-2)
        if cols == x.shape[-1]:
            return scale.unsqueeze(-1)
        raise ValueError(
            f"1D scale with shape {scale.shape} cannot be broadcast to x with shape "
            f"{x.shape}, group_shape={(rows, cols)}")
    return scale


def _group_broadcast(t, shape):
    for i, s in enumerate(shape):
        t_dim_size = t.shape[i] if i < t.ndim else 1
        if t_dim_size != s and t_dim_size != 1:
            assert s % t_dim_size == 0
            t = (t.unsqueeze(i + 1)
                 .expand(*t.shape[:i + 1], s // t_dim_size, *t.shape[i + 1:])
                 .flatten(i, i + 1))
    return t


def v1_vllm_scaled_dequant(x_q, x_s, group_shape=None, out_dtype=torch.float32):
    scale = _prep_scale(x_s, x_q, group_shape)
    if group_shape is not None:
        assert scale.shape[-1] == x_q.shape[-1] // group_shape[1]
        assert scale.shape[-2] == x_q.shape[-2] // group_shape[0]
    scale = _group_broadcast(scale.to(torch.float32), x_q.shape)
    return (x_q.to(torch.float32) * scale).to(out_dtype)
