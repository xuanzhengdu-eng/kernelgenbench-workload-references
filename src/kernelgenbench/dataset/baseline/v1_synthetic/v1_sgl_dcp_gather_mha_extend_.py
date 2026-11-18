"""SGLang DCP MHA extend gather (mha_extend mode).

Official source (pinned commit):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/dcp/comm.py#L204-L256

``all_gather_kv_cache_for_mha_extend`` first gathers the cached prefix rows with
``all_gather_kv_cache_for_dcp``, splits the packed nope/RoPE columns, aligns the
prefix dtype to the current extend tensors (torch.cat cannot promote fp8 to
bf16), then re-orders both parts request by request as
``prefix rows of request r`` followed by ``extend rows of request r``.  The
returned pair is contiguous.
"""
import torch

from .v1_sgl_dcp_gather_cache_ import v1_sgl_dcp_gather_cache_


def v1_sgl_dcp_gather_mha_extend_(prefix_kv_a, prefix_k_pe, extend_prefix_lens,
                                  extend_seq_lens, extend_kv_a, extend_k_pe,
                                  group=None, local_indices=None):
    if local_indices is not None:
        prefix_kv_a = prefix_kv_a[local_indices.long()]
        prefix_k_pe = prefix_k_pe[local_indices.long()]
    gathered = v1_sgl_dcp_gather_cache_(
        prefix_kv_a, prefix_k_pe, extend_prefix_lens, group=group)
    prefix_kv_a, prefix_k_pe = gathered.split(
        [extend_kv_a.shape[-1], extend_k_pe.shape[-1]], dim=-1)
    prefix_kv_a = prefix_kv_a.squeeze(1)
    if prefix_kv_a.dtype != extend_kv_a.dtype:
        prefix_kv_a = prefix_kv_a.to(extend_kv_a.dtype)
    if prefix_k_pe.dtype != extend_k_pe.dtype:
        prefix_k_pe = prefix_k_pe.to(extend_k_pe.dtype)
    prefix_lens_cu = torch.zeros(len(extend_prefix_lens) + 1, dtype=torch.int32)
    extend_lens_cu = torch.zeros_like(prefix_lens_cu)
    prefix_lens_cu[1:] = torch.cumsum(extend_prefix_lens, dim=0)
    extend_lens_cu[1:] = torch.cumsum(extend_seq_lens, dim=0)
    kv_a_tuple = ()
    k_pe_tuple = ()
    for i in range(len(extend_prefix_lens)):
        kv_a_tuple += (prefix_kv_a[prefix_lens_cu[i]:prefix_lens_cu[i + 1]],
                       extend_kv_a[extend_lens_cu[i]:extend_lens_cu[i + 1]])
        k_pe_tuple += (prefix_k_pe[prefix_lens_cu[i]:prefix_lens_cu[i + 1]],
                       extend_k_pe[extend_lens_cu[i]:extend_lens_cu[i + 1]])
    return (torch.cat(kv_a_tuple, dim=0).contiguous(),
            torch.cat(k_pe_tuple, dim=0).contiguous())
