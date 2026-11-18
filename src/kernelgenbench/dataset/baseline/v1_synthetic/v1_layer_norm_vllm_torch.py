"""vLLM ``LayerNorm`` native contract, 723 row 52 branch ``backend="vllm"``.

Official source (pinned commit f358f0578a83f4e1fe6fc49a492d3ae9362de8e5):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/layernorm.py#L221-L292

Contract kept from the old reference (``LayerNorm.forward_native``):
  * ``gamma`` (weight) and ``beta`` (bias) may be ``None`` independently; the
    layer only normalizes then applies whichever affine tensors exist.
  * the compute dtype is ``vals.dtype`` when ``gamma`` is ``None`` and otherwise
    ``torch.promote_types(vals.dtype, gamma.dtype)``; the result is cast back to
    the input dtype with ``type_as`` (``LayerNorm._compute_dtype`` +
    ``forward_native``).
  * ``normalized_shape`` is the last input axis; arbitrary leading dimensions.
  * at this commit ``forward_cuda`` delegates to ``forward_native``.

The real pinned operator is the ``LayerNorm`` module in the vLLM site; the test
constructs it and binds ``forward_cuda`` (which routes to this same native body)
as the Solution.
"""
import torch
import torch.nn.functional as F


def v1_layer_norm_vllm_torch(vals, gamma, beta, epsilon):
    compute_dtype = vals.dtype if gamma is None else torch.promote_types(
        vals.dtype, gamma.dtype)
    if compute_dtype != vals.dtype:
        return F.layer_norm(
            vals.to(compute_dtype),
            (vals.shape[-1],),
            gamma,
            beta,
            epsilon,
        ).type_as(vals)
    return F.layer_norm(vals, (vals.shape[-1],), gamma, beta, epsilon)
