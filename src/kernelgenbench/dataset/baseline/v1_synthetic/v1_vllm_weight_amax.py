# v1_vllm_weight_amax: vLLM weight_amax; 723-row 405.
"""``max(|weight|)`` from a single aminmax pass.

Source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/quant_utils.py#L40-L45

Contract
--------
``weight`` is a real floating tensor with a nonempty reduction domain. The
result is ``maximum(|amin|, |amax|)`` over ``dim`` (``None`` means every
element) with ``keepdim`` honoured, computed without materialising a
full-size ``abs()``. Output dtype and device follow the input; NaN and
``-inf`` inputs propagate (an all-finite tensor can still return ``inf`` when a
single element is ``-inf``); a zero tensor returns ``0.``. Neither the input
nor its shape is modified.

Two behaviours are pinned by the arguments being keyword-only and by the
reduction being a single ``aminmax`` call: calling it with positional
``dim``/``keepdim`` is a ``TypeError``, and an empty reduction domain raises
``RuntimeError`` instead of returning an identity, because aminmax has no
identity element.
"""
import torch


def v1_vllm_weight_amax(weight, *, dim=None, keepdim=False):
    lo, hi = weight.aminmax(dim=dim, keepdim=keepdim)
    return torch.maximum(lo.abs(), hi.abs())
