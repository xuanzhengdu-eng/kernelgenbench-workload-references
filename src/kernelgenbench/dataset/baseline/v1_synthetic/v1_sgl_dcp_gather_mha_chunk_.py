"""SGLang DCP MHA chunk-extend gather (mha_chunk mode).

Official source (pinned commit):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/dcp/comm.py#L185-L202

``all_gather_kv_cache_for_mha_chunk_extend`` gathers the prefix rows (with the
``[rows, 1, width]`` chunk layout) and returns a two-tensor tuple: the nope part
with the singleton chunk axis squeezed out, and the RoPE part kept as
``[rows, 1, rope_dim]``.  Both are contiguous.  When DCP is disabled the inputs
are returned as contiguous views of the same storage.
"""
import torch

from .v1_sgl_dcp_gather_cache_ import v1_sgl_dcp_gather_cache_


def v1_sgl_dcp_gather_mha_chunk_(kv_a, k_pe, seq_lens_cpu, group=None,
                                 prefix_starts_cpu=None, dcp_enabled=True):
    assert k_pe is not None and k_pe.ndim == 3 and k_pe.shape[1] == 1
    assert kv_a.shape[0] == k_pe.shape[0]
    if not dcp_enabled:
        return kv_a.contiguous(), k_pe.contiguous()
    gathered = v1_sgl_dcp_gather_cache_(
        kv_a.unsqueeze(1), k_pe, seq_lens_cpu, group=group,
        prefix_starts_cpu=prefix_starts_cpu, interleave_size=1).squeeze(1)
    width = kv_a.shape[-1]
    return (gathered[:, :width].contiguous(),
            gathered[:, width:].unsqueeze(1).contiguous())
