# Source: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/models/deepseek_v2.py#L1173-L1346
"""Single-rank unquantized dual-stream MoE with a separate shared expert."""
# Source: https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py#L598-L628
import torch
import torch.nn.functional as F


def v1_sgl_dual_stream_routed_shared(layer, hidden_states):
    assert not layer.experts.moe_runner_config.inplace

    def mlp(x, w13, w2):
        gate, up = F.linear(x, w13).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, w2)

    logits = F.linear(hidden_states, layer.gate.weight)
    probabilities, indices = logits.softmax(-1).topk(layer.config.num_experts_per_tok, dim=-1)
    if layer.config.norm_topk_prob:
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    shared = layer.shared_experts
    result = mlp(hidden_states, shared.gate_up_proj.weight, shared.down_proj.weight)
    # Triton tl.dot on fp32 uses TF32 operands, then rounds each GEMM result to fp16.
    def tf32(values):
        return (values.contiguous().view(torch.int32) & -8192).view(torch.float32)
    for expert in range(layer.config.n_routed_experts):
        tokens, lanes = (indices == expert).nonzero(as_tuple=True)
        gate_up = F.linear(tf32(hidden_states[tokens]), tf32(layer.experts.w13_weight[expert])).half().float()
        gate, up = gate_up.chunk(2, dim=-1)
        routed = F.linear(tf32(F.silu(gate) * up), tf32(layer.experts.w2_weight[expert]))
        weighted = (routed * probabilities[tokens, lanes, None]).half().float()
        result.index_add_(0, tokens, weighted)
    return result
