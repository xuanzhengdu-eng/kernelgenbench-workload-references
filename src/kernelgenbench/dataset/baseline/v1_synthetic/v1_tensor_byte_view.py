"""DeepSpeed byte reinterpretation of a contiguous tensor.

Official source (DeepSpeed, commit 8af59b2e7a91c227081617240e98aa282eadc5f9):
https://github.com/deepspeedai/DeepSpeed/blob/8af59b2e7a91c227081617240e98aa282eadc5f9/csrc/utils/tensor_cast.cpp#L7-L20
Exposed to Python as ``UtilsBuilder().load().cast_to_byte_tensor`` (csrc/utils/py_ds_utils.cpp).

Value and data-pointer contract only: a tensor with at most one byte of payload is
returned unchanged; otherwise the result is a flat ``uint8`` tensor of ``nbytes``
elements aliasing the source data pointer.  The upstream C++ builds a non-owning
``at::from_blob`` view with distinct storage, while this torch view keeps the source
alive through shared storage; callers must keep the input alive either way.
"""
import torch


def v1_tensor_byte_view(src_tensor):
    assert src_tensor.is_contiguous()
    if src_tensor.numel() * src_tensor.element_size() <= 1:
        return src_tensor
    return src_tensor.view(torch.uint8).reshape(-1)
