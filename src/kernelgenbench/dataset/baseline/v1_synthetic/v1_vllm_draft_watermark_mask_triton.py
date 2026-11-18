"""vLLM draft watermark mask, CUDA Triton backend reference.

https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/watermark.py#L407-L441

The pinned CUDA branch launches ``_repeated_context_mask_kernel``; this eager
transcription keeps its observable contract: a padded row (req_index < 0)
returns False without touching ``prior_contexts``, and the history comparison
uses only the window ``[step - max_history, step)``.  The test binds the real
compiled Triton function.
"""
import torch
from .v1_vllm_repeated_context_mask import v1_vllm_repeated_context_mask

def v1_vllm_draft_watermark_mask_triton(all_token_ids,req_indices,prompt_lens,total_lens,prior_contexts,contexts,steps,enabled,max_history,include_prompt):
    base=v1_vllm_repeated_context_mask(all_token_ids,req_indices,prompt_lens,total_lens,contexts,max_history,include_prompt,False,steps)
    out=[];gpu=contexts.device.type!='cpu'
    for row in range(len(contexts)):
        req=int(req_indices[row]);step=int(steps[row]);ctx=contexts[row].tolist()
        if gpu and req<0:
            out.append(False);continue
        repeated=bool(base[row]);start=0 if max_history is None else max(0,step-max_history)
        for i in range(start,step):
            if prior_contexts[row,i].tolist()==ctx:repeated=True;break
        if include_prompt and any(x<0 for x in ctx):repeated=True
        prior_contexts[row,step].copy_(contexts[row])
        out.append(bool(enabled[row]) and not repeated)
    return torch.tensor(out,dtype=torch.bool,device=contexts.device)
