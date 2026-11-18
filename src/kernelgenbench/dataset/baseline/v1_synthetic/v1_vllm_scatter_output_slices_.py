"""Scatter consecutive token slices into a list/dict; optional independent copies.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/models/utils.py#L1397-L1411
"""
def v1_vllm_scatter_output_slices_(output,indices,per_item_out_tokens,dest,clone=False):
    offset=0
    for idx in indices:
        n=per_item_out_tokens[idx]
        value=output[offset:offset+n]
        dest[idx]=value.clone() if clone else value
        offset+=n
