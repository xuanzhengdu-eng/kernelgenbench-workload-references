"""SGLang sequence-parallel entry scatter with the explicit token-count state.

Source (pinned commit):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/layernorm_sp.py#L93-L129

Contract: shards the replicated ``[M, h]`` hidden states along the token dim.
The tensor is padded with zero rows up to a multiple of ``tp_size`` and each
rank takes ``ceil(M / tp_size)`` rows with ``tensor_split``, made contiguous.
The real (unpadded) token count is stored in the module state and returned here
as the second element, mirroring ``set_sp_num_tokens``/``sp_num_tokens``.  With
``tp_size == 1`` the input object is returned unchanged (no copy, no pad).  The
input is never modified.  ``tp_size``/``tp_rank`` replace the runtime global
parallel state so the function can be driven directly.
"""
import torch.nn.functional as F


def v1_sgl_sp_entry_scatter(hidden_states, tp_size, tp_rank):
    num_tokens = hidden_states.shape[0]
    if tp_size == 1:
        return hidden_states, num_tokens
    padded = (num_tokens + tp_size - 1) // tp_size * tp_size
    if padded != num_tokens:
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded - num_tokens))
    return hidden_states.tensor_split(tp_size)[tp_rank].contiguous(), num_tokens
