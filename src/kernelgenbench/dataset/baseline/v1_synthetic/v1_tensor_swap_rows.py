"""Megatron in-place row swap with simultaneous-assignment semantics.

Official source:
https://github.com/NVIDIA/Megatron-LM/blob/266f1c97dca477ca8b92c16087506da2000b0b84/megatron/core/inference/utils.py#L232-L236

``tensor_swap`` swaps ``x[src_idxs]`` and ``x[dst_idxs]`` with one Python
tuple assignment, so both gathers are materialised before either scatter.
That detail is observable when ``src_idxs`` and ``dst_idxs`` overlap.
The function returns ``None`` and mutates ``x`` in place.
"""
def v1_tensor_swap_rows(x,src_idxs,dst_idxs):
    src=x[src_idxs].clone();dst=x[dst_idxs].clone()
    x[dst_idxs]=src;x[src_idxs]=dst
    return None
