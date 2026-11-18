"""DeepSpeed Random-LTD token gather (forward of GatherTokens).

Official source (deepspeedai/DeepSpeed, commit
8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/random_ltd/dropping_utils.py#L82-L103
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/random_ltd/pt_binding.cpp#L26-L81

Activations are [N, S, C] with batch_first=True or [S, N, C] otherwise;
sorted_indices is [N, retained] int32 and selects rows of the sequence axis.
The entry returns ``(activations, gathered)``: the first element is the caller's
own tensor (no copy), the second a fresh [N, retained, C] / [retained, N, C]
tensor. The CUDA kernel copies 16-byte chunks, so the channel dim must be dense
and a multiple of 16/element_size; it only dispatches float32 and float16.
"""
import torch


def _rows(tensor, batch_first):
    return tensor if batch_first else tensor.transpose(0, 1)


def v1_random_ltd_gather(activations, sorted_indices, batch_first):
    channels = activations.shape[2]
    assert sorted_indices.dtype == torch.int32
    assert activations.dtype in (torch.float32, torch.float16)
    assert activations.stride(-1) == 1
    assert channels % (16 // activations.element_size()) == 0
    rows = _rows(activations, batch_first)
    retained = sorted_indices.shape[1]
    assert sorted_indices.shape[0] == rows.shape[0] and retained <= rows.shape[1]

    index = sorted_indices.to(torch.int64).unsqueeze(-1).expand(-1, -1, channels)
    gathered = torch.gather(rows, 1, index)
    if not batch_first:
        gathered = gathered.transpose(0, 1).contiguous()
    return activations, gathered
