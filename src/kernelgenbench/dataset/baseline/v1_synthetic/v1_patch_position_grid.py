"""SGLang Pixtral per-image flattened row/column patch coordinates.

https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/models/pixtral.py#L453-L469
Coordinates are CPU int64 for CPU and GPU inputs: the upstream function builds
them with torch.arange/torch.meshgrid without a device argument.
"""
import torch


def v1_patch_position_grid(patch_embeds_list):
    assert patch_embeds_list, "at least one patch grid is required"
    pairs = [
        (h, w)
        for patch in patch_embeds_list
        for h in range(patch.shape[-2])
        for w in range(patch.shape[-1])
    ]
    return torch.tensor(pairs, dtype=torch.int64, device="cpu").reshape(-1, 2)
