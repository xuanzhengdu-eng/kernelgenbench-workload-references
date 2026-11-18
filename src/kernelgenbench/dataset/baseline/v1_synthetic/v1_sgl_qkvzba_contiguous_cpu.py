"""SGLang CPU QKVZ/BA split for the contiguous layout.

Source:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/model/qwen3.cpp#L67-L97
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/model/qwen3.cpp#L200-L241

Contiguous input layout is ``[all Q | all K | all V | all Z]`` and
``[all B | all A]``.  Unlike the grouped entry, no head-group ratio or
equal head-width check is applied by the native code.
"""
import torch


def v1_sgl_qkvzba_contiguous_cpu(
    mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
):
    batch = mixed_qkvz.shape[0]
    qkv_dim = 2 * num_heads_qk * head_qk + num_heads_v * head_v
    v_dim = num_heads_v * head_v

    assert mixed_qkvz.shape == (batch, qkv_dim + v_dim)
    assert mixed_ba.shape == (batch, 2 * num_heads_v)

    # The native code copies the first qkv_dim columns into a fresh tensor.
    mixed_qkv = mixed_qkvz[:, :qkv_dim].contiguous()
    z = mixed_qkvz[:, qkv_dim:].reshape(
        batch, num_heads_v, head_v
    ).contiguous()
    b = mixed_ba[:, :num_heads_v].contiguous()
    a = mixed_ba[:, num_heads_v:].contiguous()
    return mixed_qkv.clone(), z.clone(), b.clone(), a.clone()
