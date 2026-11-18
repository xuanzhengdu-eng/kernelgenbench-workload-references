"""DeepSpeed inference Q/K/V padding to head widths 32/64/128 and KV length 128.

Official source (deepspeedai/DeepSpeed, commit
8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/pt_binding.cpp#L1237-L1277
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/transformer/inference/csrc/gelu.cu#L539-L596

The C++ add_padding<T> pads query to its own sequence length and key/value to
128 positions. The padded head width is 32 for head_size<32, 64 for
head_size<64, otherwise 128. Padding elements are zero. Only FP16 and BF16
kernels are instantiated with a real body (the FP32 pad_head_seq overload is
empty), so this reference accepts those two dtypes and exposes the same
preconditions used by the launcher.
"""
import torch


def v1_ds_pad_qkv(query, key, value):
    batch, heads, query_seq, head_size = query.shape
    kv_seq = key.shape[2]
    assert head_size % 8 == 0 and head_size <= 128
    assert query_seq % 16 == 0
    assert kv_seq <= 128
    assert query.dtype in (torch.float16, torch.bfloat16)
    assert query.dtype == key.dtype == value.dtype
    assert key.shape == value.shape == (batch, heads, kv_seq, head_size)

    padded_head = 32 if head_size < 32 else 64 if head_size < 64 else 128
    outputs = []
    for source, padded_seq in ((query, query_seq), (key, 128), (value, 128)):
        padded = torch.zeros(
            (batch, heads, padded_seq, padded_head),
            device=source.device,
            dtype=source.dtype,
        )
        padded[:, :, :source.shape[2], :head_size] = source
        outputs.append(padded)
    return outputs
