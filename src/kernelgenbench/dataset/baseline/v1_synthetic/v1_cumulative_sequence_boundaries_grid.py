"""SGLang cumulative sequence boundaries from a CPU vision grid.

Official source (sgl-project/sglang, commit
62384de1ae5e3bbd3159afd1639525b0bf39fbf9):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/models/utils.py#L478-L495

``compute_cu_seqlens_from_grid_numpy`` expects a CPU ``[T, 3]`` grid with
columns ``[repeat, H, W]``, repeats each ``H * W`` by ``repeat`` and returns the
leading-zero cumulative sum as an int32 CPU tensor. It goes through NumPy, so
the result is a CPU tensor even when the input grid is int64.
"""
import numpy as np
import torch


def v1_cumulative_sequence_boundaries_grid(grid_thw):
    assert grid_thw.device.type == "cpu"
    assert grid_thw.ndim == 2 and grid_thw.shape[1] == 3
    arr = grid_thw.numpy()
    cu_seqlens = np.repeat(arr[:, 1] * arr[:, 2], arr[:, 0]).cumsum(
        axis=0, dtype=np.int32
    )
    cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
    return torch.from_numpy(cu_seqlens)
