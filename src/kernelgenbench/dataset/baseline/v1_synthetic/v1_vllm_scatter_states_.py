"""Scatter unique request state rows in-place.

Pinned source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/mamba/ops/scatter_states.py#L69-L106

state is a dense device tensor, src has the same trailing shape and one row per index,
and indices is a 1-D int32/int64 tensor.  The rows are written into state in place
(unique indices; the source kernel is non-atomic) and the call returns None.  src and
indices are not modified.
"""
def v1_vllm_scatter_states_(state,src,indices):
    state.index_copy_(0,indices.long(),src.to(state.dtype))
    return None
