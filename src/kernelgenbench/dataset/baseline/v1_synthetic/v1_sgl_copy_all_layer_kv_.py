"""Copy arbitrary overlapping slot rows in every KV buffer.

Official entry actually executed for this item (CUDA Triton backend):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/kvcache/cache_move.py#L181-L209
The 2-D tiled kernel it launches (gather-then-scatter, in-place safe):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/kvcache/cache_move.py#L145-L178

CPU counterpart of the same entry (AOT kernel; not runnable on this node, see
the report row — the installed sgl_kernel wheel does not register the op):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/csrc/cpu/kvcache.cpp#L144-L184
and its Python wrapper:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/aot/python/sgl_kernel/kvcacheio.py#L350-L361

Every backend gathers (stages) each buffer's source rows before writing any
target row, so overlapping src/tgt lists are safe; repeated target slots resolve
in listed order on the CPU loop (the parallel Triton kernel leaves them
undefined, so the tests use unique targets).  Inputs are a list of KV buffers
[slots, byte_width_i], a target slot vector and a source slot vector of equal
length and matching int32/int64 dtype.  Buffers are mutated in place (the same
objects are kept) and None is returned.
"""
def v1_sgl_copy_all_layer_kv_(buffers,tgt_loc,src_loc):
    for buf in buffers:
        staged=buf.index_select(0,src_loc.long()).clone()
        for i,dst in enumerate(tgt_loc.tolist()):buf[dst].copy_(staged[i])
