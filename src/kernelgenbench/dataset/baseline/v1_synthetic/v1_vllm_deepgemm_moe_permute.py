# Source scope: vLLM fixed commit deepgemm_moe_permute allocation/layout selection and ep_scatter call; the called kernel performs expert grouping, padding, scale packing and inverse-index writes. Local CPU-count model only covers block_size 32/128, 128-row alignment and listed scale layouts; no parity claim.
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py#L467-L559
# Exact range: def deepgemm_moe_permute L467 through the 5-tuple return at L559;
# deepgemm_unpermute_and_reduce starts at L562 and is not part of this item.
"""Expert-local DeepGEMM activation permutation, including scale packing.

The chosen contract fixes a 128-row expert alignment and uses exact
CPU token counts. Padding payload and inverse indices for invalid routes are
unspecified. Parallel implementations may choose any order within an expert.
"""
import torch

def v1_vllm_deepgemm_moe_permute(aq, aq_scale, topk_ids, local_num_experts,
                                  expert_map=None, block_size=128,
                                  use_psum_layout=False):
    assert aq.ndim == topk_ids.ndim == aq_scale.ndim == 2
    m, h = aq.shape
    assert m > 0 and topk_ids.shape[0] == m and topk_ids.shape[1] > 0
    assert topk_ids.dtype in (torch.int32, torch.int64) and h % block_size == 0
    assert block_size in (32, 128) and local_num_experts > 0
    assert aq_scale.shape == (m, h // block_size)
    assert aq_scale.dtype in (torch.float32, torch.uint8)
    assert all(x.device == aq.device for x in (aq_scale, topk_ids))
    mapping = None if expert_map is None else expert_map.cpu().tolist()
    if expert_map is not None:
        assert expert_map.ndim == 1 and expert_map.device == aq.device
    routes = topk_ids.cpu().tolist()
    groups = [[] for _ in range(local_num_experts)]
    for t, row in enumerate(routes):
        for k, ex in enumerate(row):
            assert ex >= -1
            if ex == -1: continue
            if mapping is not None:
                assert ex < len(mapping)
                ex = mapping[ex]
            if ex < 0: continue
            assert ex < local_num_experts
            groups[ex].append((t, k))
    align = 128
    lens = [len(g) for g in groups]
    padded = [((n + align - 1) // align) * align for n in lens]
    size = sum(padded)
    out = torch.empty((size, h), dtype=aq.dtype, device=aq.device)
    if aq_scale.dtype == torch.uint8:
        ns = (aq_scale.shape[1] + 3) // 4
        scale = torch.empty_strided((size, ns), (1, ((size + 3)//4)*4), dtype=torch.int32, device=aq.device)
    else:
        scale = torch.zeros((size, aq_scale.shape[1]), dtype=torch.float32, device=aq.device)
    layout = (torch.empty((local_num_experts,), dtype=torch.int32, device=aq.device)
              if use_psum_layout else torch.full((size,), -1, dtype=torch.int32, device=aq.device))
    inv = torch.empty(topk_ids.shape, dtype=torch.int32, device=aq.device)
    start = 0
    for ex, group in enumerate(groups):
        if use_psum_layout: layout[ex] = start + len(group)
        else: layout[start:start+len(group)] = ex
        for pos, (t, k) in enumerate(group, start):
            out[pos] = aq[t]
            if aq_scale.dtype == torch.uint8:
                s = aq_scale[t].to(torch.int64)
                for p in range(scale.shape[1]):
                    value = sum((int(s[4*p+j]) if 4*p+j < len(s) else 0) << (8*j) for j in range(4))
                    scale[pos, p] = value if value < 2**31 else value-2**32
            else: scale[pos] = aq_scale[t]
            inv[t, k] = pos
        start += padded[ex]
    return out, scale, layout, inv, align
