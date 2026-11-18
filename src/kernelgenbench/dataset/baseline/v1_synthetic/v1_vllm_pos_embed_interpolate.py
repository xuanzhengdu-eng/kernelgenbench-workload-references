"""vLLM bilinear position lookup with spatial-merge order and temporal repetition.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/models/qwen3_vl.py#L304-L374
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/models/cohere_compass.py#L271-L341
Returns (t * h * w, hidden) in spatial-merge order; the same formula is copied
in the two upstream model files.
"""
import torch


def v1_vllm_pos_embed_interpolate(embed_weight, t, h, w, num_grid_per_side, m_size, dtype):
    assert h % m_size == 0 and w % m_size == 0
    hidden_dim = embed_weight.shape[1]
    device = embed_weight.device
    h_idxs = torch.linspace(0, num_grid_per_side - 1, h, dtype=torch.float32, device=device)
    w_idxs = torch.linspace(0, num_grid_per_side - 1, w, dtype=torch.float32, device=device)
    h_floor = h_idxs.to(torch.long)
    w_floor = w_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)
    dh = h_idxs - h_floor
    dw = w_idxs - w_floor
    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")
    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01
    h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
    w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
    indices = (h_grid * num_grid_per_side + w_grid).reshape(4, -1)
    weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1).to(dtype=dtype)
    embeds = embed_weight[indices]
    embeds *= weights
    combined = embeds.sum(dim=0)
    combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
    combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
    repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
    return repeated.to(dtype=dtype)
