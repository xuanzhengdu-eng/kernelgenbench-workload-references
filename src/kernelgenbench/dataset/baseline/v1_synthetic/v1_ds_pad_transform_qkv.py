"""DeepSpeed inference Q/K/V transform and padding (``pad_transform_{fp16,bf16}``).

Source: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/pt_binding.cpp#L1279-L1325
Kernel: https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/transform.cu#L293-L388

``[B, S, H*D]`` projections become ``[B, H, S, D]`` per Q/K/V. With
``add_padding`` the head width is rounded up to 32/64/128 (by ``D < 32``,
``D < 64``, else 128) and K/V are padded to a 128-token sequence; without it the
head width and K/V sequence stay as given. The CUDA kernel is only instantiated
for fp16 and bf16 (the fp32 entry points are empty stubs), so the reference keeps
that restriction. The upstream binding returns three views into one shared
inference workspace, so a later call overwrites earlier results; callers consume
them immediately, and this reference returns fresh tensors instead.
"""
import torch


def v1_ds_pad_transform_qkv(query, key, value, heads, add_padding):
    batch, q_len, width = query.shape
    head_dim = width // heads
    assert head_dim % 8 == 0 and head_dim <= 128 and q_len % 2 == 0
    assert add_padding or key.shape[1] % 2 == 0
    assert not add_padding or key.shape[1] <= 128
    assert query.dtype in (torch.float16, torch.bfloat16)
    assert query.dtype == key.dtype == value.dtype
    padded = d = head_dim
    if add_padding:
        padded = 32 if head_dim < 32 else 64 if head_dim < 64 else 128
    kv_len = 128 if add_padding else key.shape[1]
    out = []
    for source, seq in ((query, q_len), (key, kv_len), (value, kv_len)):
        target = torch.zeros((batch, heads, seq, padded), device=source.device,
                             dtype=source.dtype)
        target[:, :, :source.shape[1], :d] = source.reshape(
            batch, source.shape[1], heads, d).transpose(1, 2)
        out.append(target)
    return out
