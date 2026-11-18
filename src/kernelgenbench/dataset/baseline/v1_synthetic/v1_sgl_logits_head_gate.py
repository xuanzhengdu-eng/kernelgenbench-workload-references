"""BF16 projection to FP32 then query-scaled DSA head logits.
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/attention/dsa/head_gate.py#L53-L63
"""
def v1_sgl_logits_head_gate(x,weight,n_heads_inv_sqrt,softmax_scale,q_scale):
    weights=(x.float()@weight.float().transpose(0,1))*n_heads_inv_sqrt
    return weights.unsqueeze(-1)*q_scale*softmax_scale
