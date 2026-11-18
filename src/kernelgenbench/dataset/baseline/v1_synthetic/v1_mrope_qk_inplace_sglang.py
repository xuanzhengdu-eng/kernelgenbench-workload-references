"""Multimodal T/H/W rotary application to Q/K in place (SGLang branch).

CUDA dispatch and fused launcher:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/rotary_embedding/mrope.py#L270-L291
Triton kernel, including the GLM axis map:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/attention/rotary_triton.py#L12-L153

Inputs are [tokens, heads*head_size] Q/K, [3, tokens] T/H/W positions and a
same-dtype [max_position, rotary_dim] cache holding cosine then sine halves.
The official entry mutates Q/K and returns the same tensor objects.
"""
import torch


def _axis_map(section, interleaved, glm):
    half = sum(section)
    if glm:
        assert interleaved
        spent = [0, 0, 0]
        axes = []
        for lane in range(half):
            axis = lane % 3
            while spent[axis] >= section[axis]:
                axis = (axis + 1) % 3
            spent[axis] += 1
            axes.append(axis)
        return axes
    if interleaved:
        axes = [0] * half
        for axis in (1, 2):
            for lane in range(axis, min(3 * section[axis], half), 3):
                axes[lane] = axis
        return axes
    return [axis for axis, size in enumerate(section) for _ in range(size)]


def v1_mrope_qk_inplace_sglang(positions, query, key, cos_sin_cache, head_size,
                              rotary_dim, mrope_section, mrope_interleaved=False,
                              mrope_interleaved_glm=False, is_neox_style=True):
    assert positions.ndim == 2 and positions.shape[0] == 3
    assert sum(mrope_section) == rotary_dim // 2
    assert query.ndim == key.ndim == 2
    assert query.shape[0] == key.shape[0] == positions.shape[1]
    assert query.dtype == key.dtype == cos_sin_cache.dtype
    assert query.device == key.device == cos_sin_cache.device == positions.device
    assert len(mrope_section) == 3 and min(mrope_section) >= 0
    assert 0 < rotary_dim <= head_size and rotary_dim % 2 == 0
    assert query.stride(-1) == key.stride(-1) == positions.stride(-1) == 1
    assert query.shape[1] % head_size == 0 and key.shape[1] % head_size == 0
    if query.shape[0] == 0:
        return query, key
    if mrope_interleaved:
        assert rotary_dim == head_size and (head_size & (head_size - 1)) == 0
    half = rotary_dim // 2
    axes = _axis_map(mrope_section, mrope_interleaved, mrope_interleaved_glm)
    axis = torch.tensor(axes, device=positions.device, dtype=torch.long)
    lanes = torch.arange(half, device=positions.device)
    selected = positions[axis, :].transpose(0, 1).long()
    cos = cos_sin_cache[selected, lanes].float().unsqueeze(1)
    sin = cos_sin_cache[selected, lanes + half].float().unsqueeze(1)

    def apply(x):
        v = x.view(x.shape[0], -1, head_size)
        if is_neox_style:
            a = v[..., :half].float()
            b = v[..., half:rotary_dim].float()
            first = (a * cos - b * sin).to(x.dtype)
            second = (b * cos + a * sin).to(x.dtype)
            v[..., :half] = first
            v[..., half:rotary_dim] = second
        else:
            a = v[..., :rotary_dim:2].float()
            b = v[..., 1:rotary_dim:2].float()
            first = (a * cos - b * sin).to(x.dtype)
            second = (b * cos + a * sin).to(x.dtype)
            v[..., :rotary_dim:2] = first
            v[..., 1:rotary_dim:2] = second

    apply(query)
    apply(key)
    return query, key
