"""SGLang vision-attention dummy-head padding (``pad_vit_attn_dummy_heads``).

Source: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/attention/vision_utils.py#L26-L67

Called while loading a ViT checkpoint when ``num_attention_heads`` is not
divisible by the attention-TP size: the missing heads are zero-filled so every
rank owns the same number of heads. ``qkv_proj`` splits its first axis into the
three fused projections, pads each with ``num_dummy_heads * head_dim`` rows and
re-concatenates them; separate ``q/k/v`` projections pad their first axis;
``proj.weight`` pads the last axis; ``q_norm``/``k_norm`` pad their flat length
unless the weight already has exactly ``head_dim`` elements. Any other name is
returned unchanged, and ``num_dummy_heads == 0`` short-circuits the whole
function. Weight names that are not ``.weight``/``.bias`` raise ``RuntimeError``
on the padded branches.
"""
import torch


def v1_sgl_vit_dummy_head_pad(loaded_weight, name, head_dim, num_dummy_heads):
    if num_dummy_heads == 0:
        return loaded_weight
    extra = head_dim * num_dummy_heads

    def pad_rows(weight):
        return torch.cat([weight.unflatten(0, (-1, head_dim)),
                          weight.new_zeros(num_dummy_heads, head_dim, *weight.shape[1:])],
                         dim=0).flatten(0, 1)

    def pad_rows_vector(weight):
        return torch.cat([weight.unflatten(0, (-1, head_dim)),
                          weight.new_zeros(num_dummy_heads, head_dim)], dim=0).flatten(0, 1)

    if "attn.qkv_proj" in name:
        if name.endswith(".weight"):
            split = [part.unflatten(0, (-1, head_dim)) for part in loaded_weight.chunk(3, dim=0)]
            blocks = [torch.cat([part, part.new_zeros(num_dummy_heads, head_dim,
                                                      *loaded_weight.shape[1:])], dim=0)
                      for part in split]
            return torch.cat([block.flatten(0, 1) for block in blocks], dim=0)
        if name.endswith(".bias"):
            blocks = [torch.cat([part.unflatten(0, (-1, head_dim)),
                                 part.new_zeros(num_dummy_heads, head_dim)], dim=0)
                      for part in loaded_weight.chunk(3, dim=0)]
            return torch.cat([block.flatten(0, 1) for block in blocks], dim=0)
        raise RuntimeError(f"Unsupported weight with name={name}")
    if any(token in name for token in ("attn.q_proj", "attn.k_proj", "attn.v_proj")):
        if name.endswith(".weight") or name.endswith(".bias"):
            return pad_rows(loaded_weight) if name.endswith(".weight") else pad_rows_vector(loaded_weight)
        raise RuntimeError(f"Unsupported weight with name={name}")
    if "attn.proj.weight" in name:
        padded = loaded_weight.new_zeros(loaded_weight.shape[0], extra)
        return torch.cat([loaded_weight, padded], dim=-1)
    if "attn.q_norm.weight" in name or "attn.k_norm.weight" in name:
        if loaded_weight.numel() == head_dim:
            return loaded_weight
        return torch.cat([loaded_weight, loaded_weight.new_zeros(extra)], dim=0)
    return loaded_weight
