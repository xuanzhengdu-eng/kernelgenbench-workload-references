"""Expand SGLang frozen-KV metadata for top-k draft beams, mutating state.
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/frozen_kv_mtp_utils.py#L106-L147
"""
import torch

def v1_sgl_expand_draft_batch_(batch,topk):
    if topk==1 or batch.batch_size==0:return
    if batch.batch_size!=batch.seq_lens.shape[0]:raise RuntimeError('expected unexpanded batch')
    batch.batch_size*=topk;batch.req_pool_indices=batch.req_pool_indices.repeat_interleave(topk,dim=0);batch.seq_lens=batch.seq_lens.repeat_interleave(topk,dim=0)
    if batch.seq_lens_cpu is not None:
        batch.seq_lens_cpu=batch.seq_lens_cpu.repeat_interleave(topk,dim=0);batch.seq_lens_sum=batch.seq_lens_cpu.sum().item()
    else:batch.seq_lens_sum=batch.seq_lens.sum().item()
    batch.positions=(batch.seq_lens-1).clamp_min(0).to(torch.int64);batch.global_num_token_non_padded_cpu=batch.positions.numel()
    if batch.global_num_token_non_padded is not None:batch.global_num_token_non_padded.fill_(batch.positions.numel())
    if batch.mrope_positions is not None and batch.mrope_positions.shape[-1]*topk==batch.positions.numel():batch.mrope_positions=batch.mrope_positions.repeat_interleave(topk,dim=-1)
