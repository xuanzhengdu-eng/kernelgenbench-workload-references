"""Gather Mamba state rows; inactive requests receive zeros.

Pinned source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/mamba/ops/gather_initial_states.py#L45-L83

state is a dense device tensor, indices is a 1-D int32/int64 tensor and
has_initial_state is a 1-D bool tensor of the same length.  The result is a new tensor
with one row per index; rows whose flag is false are zero.  Reference rows may be
padded through state.stride(0) but every row itself is contiguous, and the inputs are
never mutated.
"""
import torch
def v1_vllm_gather_initial_states(state,indices,has_initial_state):
    safe=torch.where(has_initial_state,indices,torch.zeros_like(indices)).long()
    value=state.index_select(0,safe)
    mask=has_initial_state.reshape((-1,)+(1,)*(state.ndim-1))
    return torch.where(mask,value,torch.zeros_like(value))
