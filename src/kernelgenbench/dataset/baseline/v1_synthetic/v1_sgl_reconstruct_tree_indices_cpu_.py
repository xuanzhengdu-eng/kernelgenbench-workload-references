# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/spec.cpp#L689-L789
"""Reconstruct the pinned SGLang cpu tree links in caller-owned buffers."""
import torch


def v1_sgl_reconstruct_tree_indices_cpu_(tree_mask, verified_seq_len, positions, retrive_index, retrive_next_token,
            retrive_next_sibling, batch_size, draft_token_num):
    outputs = (positions, retrive_index, retrive_next_token, retrive_next_sibling)
    batch, n = batch_size, draft_token_num
    count = batch * n
    for tensor in (tree_mask, verified_seq_len, *outputs):
        assert tensor.device.type == "cpu" and tensor.is_contiguous()
    assert tree_mask.dtype == torch.bool and tree_mask.numel() == count * n
    assert verified_seq_len.dtype in (torch.int32, torch.int64) and verified_seq_len.numel() == batch
    assert all(t.dtype == verified_seq_len.dtype and t.numel() == count for t in outputs)
    mask = tree_mask.flatten()[:count * n].view(batch, n, n).bool()
    pos, index, child, sibling = (x.view(batch, n) for x in outputs)
    cols = torch.arange(n, device=mask.device)
    for row in range(batch):
        ancestors = torch.tril(mask[row], diagonal=-1)
        parent = torch.where(ancestors, cols, -1).amax(dim=1) if n else cols
        pos[row].copy_(ancestors.sum(dim=1) + verified_seq_len.flatten()[row])
        index[row].copy_(row * n + cols)
        for node in range(n):
            later = cols > node
            descendants = cols[later & ancestors[:, node]]
            peers = cols[later & (parent[node] >= 0) & (parent == parent[node])]
            child[row, node] = descendants[0] if descendants.numel() else -1
            sibling[row, node] = peers[0] if peers.numel() else -1
