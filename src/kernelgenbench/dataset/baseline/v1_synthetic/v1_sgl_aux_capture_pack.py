"""SGLang auxiliary hidden-state list/passthrough/preallocated packing.

Official sources:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/aux_hidden_states.py#L17-L102

``list`` captures borrowed storage by cloning unless ownership is transferred,
then ``pack_aux_hidden_states`` returns the single element unchanged or
concatenates along the feature axis.  ``passthrough`` is the tensor input to
``pack_aux_hidden_states`` and returns the same object.  ``packer`` writes each
capture into a preallocated ``[tokens, K * hidden]`` buffer (or a fresh buffer)
and rejects a mis-shaped or mis-typed output buffer.
"""
import torch


def v1_sgl_aux_capture_pack(captures, owned, output_buffer, strategy):
    assert len(captures) > 0 and len(captures) == len(owned)
    if strategy == 'passthrough':
        return captures[0]
    if strategy == 'list':
        parts = [x if flag else x.clone() for x, flag in zip(captures, owned)]
        return parts[0] if len(parts) == 1 else torch.cat(parts, -1)
    assert strategy == 'packer'
    width = captures[0].shape[-1]
    shape = (*captures[0].shape[:-1], width * len(captures))
    if output_buffer is None:
        out = captures[0].new_empty(shape)
    else:
        if output_buffer.shape != shape or output_buffer.dtype != captures[0].dtype:
            raise ValueError('invalid output buffer')
        out = output_buffer
    for i, x in enumerate(captures):
        out[..., i * width:(i + 1) * width].copy_(x)
    return out
