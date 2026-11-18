"""DeepSpeed Random-LTD token scatter backward (ScatterTokens.backward).

Official source (deepspeedai/DeepSpeed, commit
8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/deepspeed/ops/random_ltd/dropping_utils.py#L119-L132
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/random_ltd/pt_binding.cpp#L26-L81

The backward of ScatterTokens returns ``(out_gradients, gathered)`` where
``out_gradients`` is first made contiguous (so a contiguous input is returned
as the same object) and ``gathered`` is the Random-LTD gather of the retained
rows. Inputs are not written. As an ``autograd.Function.backward`` it returns
one slot per forward input, so the tuple is padded with ``None`` for the sorted
indices and the layout flag: ``(out_gradients, gathered, None, None)``.
"""
import torch


def _rows(tensor, batch_first):
    return tensor if batch_first else tensor.transpose(0, 1)


def v1_random_ltd_scatter_back(out_gradients, sorted_indices, batch_first):
    channels = out_gradients.shape[2]
    assert sorted_indices.dtype == torch.int32
    assert out_gradients.dtype in (torch.float32, torch.float16)
    assert out_gradients.stride(-1) == 1
    assert channels % (16 // out_gradients.element_size()) == 0

    out_gradients = out_gradients.contiguous()
    rows = _rows(out_gradients, batch_first)
    retained = sorted_indices.shape[1]
    assert sorted_indices.shape[0] == rows.shape[0] and retained <= rows.shape[1]

    index = sorted_indices.to(torch.int64).unsqueeze(-1).expand(-1, -1, channels)
    gathered = torch.gather(rows, 1, index)
    if not batch_first:
        gathered = gathered.transpose(0, 1).contiguous()
    return out_gradients, gathered, None, None
