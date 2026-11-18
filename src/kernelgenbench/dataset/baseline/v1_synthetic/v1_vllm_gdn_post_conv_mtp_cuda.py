# vLLM pinned CUDA backend of the fused GDN post-convolution MTP decode op.
# torch.ops._C.fused_gdn_decode_post_conv_mtp is the kernel dispatched on CUDA by
# fused_mtp_decode_impl(); this transcription keeps the same parameter order, in-place
# writes and return contract so the CUDA entry can be bound directly as the Solution.
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L3213-L3249
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/gdn/fused_gdn_decode_kernel.cu#L150-L375
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/gdn/fused_gdn_decode_kernel.cu#L422-L597
"""vLLM fused GDN post-convolution MTP decode, pinned CUDA kernel contract.

Same contract as the Triton port in v1_vllm_gdn_post_conv_mtp_: the accepted-token
source slot selects the starting state, q/k are L2-normalised in fp32, the gated
delta-rule recurrence writes the state after every token (slot > 0), and the
gated RMSNorm is applied into the caller's ``out`` buffer, which is returned.
A request whose source slot is unavailable (slot <= 0, or more than 8 accepted
tokens) zeroes its output rows and writes no state.
"""
import torch
import torch.nn.functional as F

def v1_vllm_gdn_post_conv_mtp_cuda(mixed_qkv,a,b,A_log,dt_bias,state_indices,cu_seqlens,num_accepted_tokens,state,output_gate,norm_weight,out,scale,norm_eps,output_gate_activation):
    """Post-convolution GDN recurrence, per-token cache writes and gated norm.

    ``state`` and ``out`` are updated in place; the caller's ``out`` buffer is
    also returned, matching the pinned Python wrapper. A request whose source
    slot is unavailable (slot <= 0, or more than MAX_FUSED_GDN_MTP_TOKENS=8
    accepted tokens) zeroes its output rows and writes no state.
    """
    hv=state.shape[1];h=(mixed_qkv.shape[1]//128-hv)//2;per=hv//h;width=state_indices.shape[1]
    for request in range(state_indices.shape[0]):
        start=int(cu_seqlens[request]);end=int(cu_seqlens[request+1]);count=end-start
        if count<=0:continue
        accepted=int(num_accepted_tokens[request]);src=int(state_indices[request,accepted-1]) if 0<accepted<=width else 0
        if src<=0 or count>8:
            out[start:end].zero_();continue
        for vh in range(hv):
            kh=vh//per;mem=state[src,vh].float().clone()
            for t in range(count):
                row=start+t;v=mixed_qkv[row].float();q=v[kh*128:(kh+1)*128];k=v[(h+kh)*128:(h+kh+1)*128];val=v[(2*h+vh)*128:(2*h+vh+1)*128]
                q=q*torch.rsqrt(q.square().sum()+1e-6)*scale;k=k*torch.rsqrt(k.square().sum()+1e-6)
                decay=torch.exp(-torch.exp(A_log[vh])*F.softplus(a[row,vh].float()+dt_bias[vh].float()));beta=torch.sigmoid(b[row,vh].float())
                mem=mem*decay;delta=(val-mem@k)*beta;mem=mem+delta[:,None]*k[None,:]
                vout=(mem@q).to(torch.bfloat16).float();gate=output_gate[row,vh].float();gate=torch.sigmoid(gate) if output_gate_activation=='sigmoid' else F.silu(gate)
                out[row,vh]=(vout*torch.rsqrt(vout.square().mean()+norm_eps)*norm_weight.float()*gate).to(torch.bfloat16)
                dst=int(state_indices[request,t]);
                if dst>0:state[dst,vh]=mem.to(state.dtype)
    return out
