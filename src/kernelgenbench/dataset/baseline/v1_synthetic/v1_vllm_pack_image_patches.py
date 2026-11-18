"""Pack variable-sized CHW images into row-major flattened patches (Pixtral).
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/models/pixtral.py#L1801-L1837

Every image is a 3-D ``(channels, H, W)`` tensor with the same channel count and
dtype; ``patch_shape`` is ``(ph, pw)`` and the H/W grid is truncated to whole
patches. The output is ``(sum(rows * cols), channels * ph * pw)`` laid out
row-major per image with each row flattened as ``(channels, ph, pw)``, plus the
per-image ``(rows, cols)`` grid. The result is a fresh contiguous tensor; the
images are not modified.
"""
import torch
import torch.nn.functional as F


def v1_vllm_pack_image_patches(images, patch_shape):
    patch_height, patch_width = patch_shape
    channels = images[0].shape[0]
    grids = []
    patches = []
    for image in images:
        assert image.ndim == 3 and image.shape[0] == channels
        _, height, width = image.shape
        rows, columns = height // patch_height, width // patch_width
        grids.append((rows, columns))
        if rows == 0 or columns == 0:
            continue  # the image is smaller than one patch: no rows are packed
        patch = F.unfold(image.unsqueeze(0), kernel_size=(patch_height, patch_width),
                         stride=(patch_height, patch_width))
        patches.append(patch.squeeze(0).transpose(0, 1))
    if not patches:
        return torch.empty((0, channels * patch_height * patch_width),
                           dtype=images[0].dtype, device=images[0].device), grids
    return torch.cat(patches, dim=0), grids
