"""DeepSpeed Random-LTD GPT token sampling and mask slicing.
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/random_ltd/dropping_utils.py#L18-L53

Pinned entry ``gpt_sample_tokens``.  Samples ``reserved_length`` token positions
per (layer, batch) row from a uniform distribution, sorts each row ascending
with the native ``token_sort_`` kernel, and slices the attention mask to the
kept tokens.  The native operator is CUDA-only, so ``device`` must be CUDA.
Returns ``(sampled_indices int32 [layers, batch, reserved], new_mask or None)``.
"""
import torch


def v1_ds_random_ltd_sampling(reserved_length, seq_length, batch_size, layers=1,
                              device="cuda", attn_mask=None):
    assert 0 < reserved_length <= seq_length
    assert batch_size > 0 and layers > 0
    assert torch.device(device).type == "cuda", "Random-LTD token_sort_ is a CUDA operator"
    probs = torch.ones((layers * batch_size, seq_length), device=device)
    sampled = torch.multinomial(probs, reserved_length)
    sampled = sampled.reshape(layers, batch_size, reserved_length).to(torch.int32)
    # The pinned operator sorts the sampled positions in place, ascending.
    sampled = sampled.sort(dim=-1).values
    if attn_mask is not None:
        new_mask = attn_mask[:, :, :reserved_length, :reserved_length]
    else:
        new_mask = None
    return sampled, new_mask
