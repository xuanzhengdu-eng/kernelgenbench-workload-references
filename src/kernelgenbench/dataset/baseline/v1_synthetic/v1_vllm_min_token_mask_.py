"""vLLM minimum-token stop-logit masking and optional all-masked restore.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/sample/logits_processor/builtin.py#L254-L288
Input [B,V] logits and indexed stop-token positions; restore rows must be
consecutive as in source. Mutates and returns same tensor object.
"""
import torch

def v1_vllm_min_token_mask_(logits,mask_rows,mask_tokens,restore_rows,restore_tokens,enabled):
    if not enabled:return logits
    if restore_rows.numel()==0:
        logits.index_put_((mask_rows,mask_tokens),torch.tensor(float('-inf'),device=logits.device,dtype=logits.dtype));return logits
    saved=logits[restore_rows,restore_tokens]
    logits.index_put_((mask_rows,mask_tokens),torch.tensor(float('-inf'),device=logits.device,dtype=logits.dtype))
    rows,inverse=torch.unique_consecutive(restore_rows,return_inverse=True)
    restore=torch.isneginf(logits[rows]).all(-1)[inverse] & torch.isfinite(saved)
    logits.index_put_((restore_rows[restore],restore_tokens[restore]),saved[restore])
    return logits
