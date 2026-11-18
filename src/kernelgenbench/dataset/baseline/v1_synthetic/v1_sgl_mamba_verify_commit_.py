"""Regular SGLang post-verify Mamba state commit into persistent caches.

SGLang source:
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/spec_utils.py#L1104-L1163
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/spec_utils.py#L1166-L1426
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py#L1606-L1711
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py#L781-L848
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py#L880-L911

State indices are resolved physical slots; src_indices_raw is None for ordinary
batch-order scratch, or explicit nonnegative stable scratch rows. Distinct
valid destination slots avoid concurrent write races. PLE sidecars, ReplaySSM
and KDA fused accept are outside this specific full regular-path contract.
"""
import torch

def v1_sgl_mamba_verify_commit_(temporal,intermediate_ssm,conv_states,intermediate_conv_window,accept_lens,accept_index,seq_lens,draft_token_num,state_indices,mamba_track_indices=None,track_interval=0,src_indices_raw=None,prepared_step_indices=None):
    bs=accept_lens.shape[0]
    if accept_index.numel()==0:return None
    if prepared_step_indices is None:
        offsets=torch.arange(bs,device=accept_lens.device,dtype=accept_lens.dtype)*draft_token_num
        rows=torch.arange(bs,device=accept_lens.device,dtype=torch.int64)
        last=accept_index[rows,(accept_lens-1).long()]-offsets
        track=None
        if mamba_track_indices is not None:
            pre=seq_lens;post=pre+accept_lens
            cross=(pre//track_interval)!=(post//track_interval)
            location=(post//track_interval*track_interval-pre-1).clamp(min=0).long()
            candidate=accept_index[rows,location]-offsets
            track=torch.where(cross,candidate,torch.full_like(candidate,-1))
    else:last,track=prepared_step_indices
    if mamba_track_indices is not None:assert track is not None
    rows=range(bs) if src_indices_raw is None else src_indices_raw.tolist()
    def scatter(dst,src,dest,steps):
        if dst.numel()==0:return
        for request,src_row in enumerate(rows):
            slot=int(dest[request]);step=int(steps[request])
            if 0<=slot<dst.shape[1] and 0<=src_row<src.shape[1] and 0<=step<src.shape[2]:
                dst[:,slot].copy_(src[:,src_row,step])
    scatter(temporal,intermediate_ssm,state_indices,last)
    if mamba_track_indices is not None:scatter(temporal,intermediate_ssm,mamba_track_indices,track)
    for dst,src in zip(conv_states,intermediate_conv_window):scatter(dst,src,state_indices,last)
    if mamba_track_indices is not None:
        for dst,src in zip(conv_states,intermediate_conv_window):scatter(dst,src,mamba_track_indices,track)
    return None
