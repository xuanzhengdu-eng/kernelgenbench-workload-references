"""DeepSpeed Random-LTD token gather backward (GatherTokens.backward).

Official source (deepspeedai/DeepSpeed, commit
8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/random_ltd/dropping_utils.py#L94-L103
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/random_ltd/pt_binding.cpp#L84-L140

The saved-index backward scatters the retained gradients back into a copy of
the full activation gradient. The official code calls ``.contiguous()`` on
``a_gradients`` first and then writes into it in place through
``token_scatter_``, returning that same tensor: for a contiguous input the
caller's own buffer is modified and returned (no copy). As an
``autograd.Function.backward`` it returns one slot per forward input, so the
result is ``(updated_gradient, None, None)`` for the sorted indices and the
layout flag.
"""
import torch


def _rows(tensor, batch_first):
    return tensor if batch_first else tensor.transpose(0, 1)


def v1_random_ltd_gather_back(a_gradients, g_gradients, sorted_indices, batch_first):
    channels = a_gradients.shape[2]
    assert sorted_indices.dtype == torch.int32
    assert a_gradients.dtype in (torch.float32, torch.float16)
    assert a_gradients.stride(-1) == 1
    assert channels % (16 // a_gradients.element_size()) == 0
    assert g_gradients.shape[2] == channels

    a_gradients = a_gradients.contiguous()
    rows = _rows(a_gradients, batch_first)
    updates = _rows(g_gradients.contiguous(), batch_first)
    retained = sorted_indices.shape[1]
    assert updates.shape == (rows.shape[0], retained, channels)

    index = sorted_indices.to(torch.int64).unsqueeze(-1).expand_as(updates)
    rows.scatter_(1, index, updates)
    return a_gradients, None, None
