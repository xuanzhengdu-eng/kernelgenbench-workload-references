"""DeepSpeed Random-LTD token scatter (forward of ScatterTokens).

Official source (deepspeedai/DeepSpeed, commit
8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/random_ltd/dropping_utils.py#L106-L118
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/random_ltd/pt_binding.cpp#L84-L140

``all_activations.clone()`` is scattered in place with the retained rows, so
the returned tensor is a fresh buffer with the selected positions overwritten
and every other position copied from ``all_activations``. Inputs are unchanged.
"""
import torch


def _rows(tensor, batch_first):
    return tensor if batch_first else tensor.transpose(0, 1)


def v1_random_ltd_scatter(all_activations, layer_activations, sorted_indices, batch_first):
    channels = all_activations.shape[2]
    assert sorted_indices.dtype == torch.int32
    assert all_activations.dtype in (torch.float32, torch.float16)
    assert all_activations.stride(-1) == 1
    assert channels % (16 // all_activations.element_size()) == 0
    assert layer_activations.shape[2] == channels

    result = all_activations.clone()
    rows = _rows(result, batch_first)
    updates = _rows(layer_activations.contiguous(), batch_first)
    retained = sorted_indices.shape[1]
    assert updates.shape == (rows.shape[0], retained, channels)

    index = sorted_indices.to(torch.int64).unsqueeze(-1).expand_as(updates)
    rows.scatter_(1, index, updates)
    return result
