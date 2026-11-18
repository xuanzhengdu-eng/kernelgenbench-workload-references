# Source-checked synthetic-workload benchmark reference.
"""OP-c4159a95fc8b (split branch): vLLM DeepGEMM unpermute and reduce.

Official source:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py#L562-L576
Triton kernel:
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py#L363-L455

Contract: `output` is overwritten in place and None is returned. `a` is
[num_permuted_rows, hidden], `topk_ids`, `topk_weights` and `inv_perm` are
[num_tokens, topk]; a slot is accumulated in fp32 when
`expert_map[topk_id] >= 0` (or `topk_id >= 0` without a map), and `topk_id == -1`
is never remapped. The ep_gather kernel reads `a` and `output` with their row
strides.
"""
import torch


def v1_vllm_deepgemm_unpermute_and_reduce(a, topk_ids, topk_weights, inv_perm,
                                          expert_map, output):
    assert a.dim() == 2 and output.dim() == 2
    num_tokens, topk = topk_ids.shape
    hidden = a.shape[1]
    assert output.shape == (num_tokens, hidden)
    assert topk_weights.shape == inv_perm.shape == topk_ids.shape
    assert topk_weights.dtype == torch.float32
    assert output.dtype == a.dtype
    for token in range(num_tokens):
        acc = torch.zeros(hidden, dtype=torch.float32, device=output.device)
        for slot in range(topk):
            expert = int(topk_ids[token, slot])
            if expert_map is not None and expert != -1:
                expert = int(expert_map[expert])
            if expert >= 0:
                acc = acc + a[int(inv_perm[token, slot])].float() * topk_weights[token, slot]
        output[token].copy_(acc.to(output.dtype))
    return None
