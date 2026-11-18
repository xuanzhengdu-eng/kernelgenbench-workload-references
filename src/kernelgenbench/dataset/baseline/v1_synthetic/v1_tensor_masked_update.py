"""Megatron indexed state update with a -1 skip sentinel.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/contexts/attention_context/triton/tensor_ops.py#L395-L475

``tensor_masked_update`` copies ``new_states[i]`` into ``states[idx[i]]`` for
every i, skipping destinations equal to -1.  ``states`` may be 2-D, 3-D or 4-D
and strided; ``idx`` is 1-D and ``new_states.shape[1:] == states.shape[1:]``.
Destinations are expected to be distinct (the native kernel does not define an order for duplicate indices).  Returns None and mutates only ``states``.
"""
def v1_tensor_masked_update(states,idx,new_states):
    assert states.ndim in (2,3,4) and idx.ndim==1 and states.shape[1:]==new_states.shape[1:]
    for i in range(idx.numel()):
        j=int(idx[i])
        if j!=-1:states[j].copy_(new_states[i])
    return None
