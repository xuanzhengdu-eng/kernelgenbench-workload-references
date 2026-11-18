"""vLLM sparse per-request logit bias, standalone PyTorch processor.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/logits_processor/builtin.py#L159-L162
logits [B,V], unique valid (row,token) pairs and float32 bias vector. Update
selected logits in place, return same tensor; disabled branch is identity.
"""
import torch

def v1_vllm_sparse_logit_bias_(logits,rows,tokens,bias,enabled=True):
    if enabled:
        for i in range(rows.numel()):logits[int(rows[i]),int(tokens[i])]+=bias[i]
    return logits
