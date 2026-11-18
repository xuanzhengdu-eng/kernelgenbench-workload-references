"""vLLM repeated watermark context history mask; CPU-semantics reference.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/watermark.py#L206-L375
"""
import torch

def v1_vllm_repeated_context_mask(all_token_ids,req_indices,prompt_lens,total_lens,contexts,max_history=None,include_prompt=False,skip_partial_context=False,history_offsets=None,local_positions=None,num_speculative_steps=0):
    if max_history is not None and max_history<1:raise ValueError('max_history must be positive or None')
    if (local_positions is None)!=(num_speculative_steps==0):raise ValueError('inconsistent speculative metadata')
    rows=len(req_indices);width=contexts.shape[-1];result=[]
    for row in range(rows):
        req=int(req_indices[row]);current=contexts[row].tolist();hit=False
        if req>=0:
            prompt=int(prompt_lens[req]);total=int(total_lens[req])
            if skip_partial_context and total-prompt<width:hit=True
            else:
                start=0 if include_prompt else prompt;history=all_token_ids[req,start:total].tolist();off=0 if history_offsets is None else int(history_offsets[row]);lower=0 if max_history is None else max(0,len(history)-max(0,max_history-off))
                for pos in range(lower,len(history)):
                    preceding=([-1]*width+history[:pos])[-width:]
                    if preceding==current:hit=True;break
            if local_positions is not None:
                limit=num_speculative_steps if max_history is None else min(num_speculative_steps,max_history)
                for off in range(1,limit+1):
                    prev=row-off
                    if prev>=0 and int(local_positions[row])>=off and int(req_indices[prev])==req and int(local_positions[row])==int(local_positions[prev])+off and contexts[prev].tolist()==current:hit=True;break
        # CPU post-processing also marks padded request rows; Triton exits early.
        if local_positions is not None and include_prompt and (contexts.device.type=="cpu" or req>=0) and any(v<0 for v in current):hit=True
        result.append(hit)
    return torch.tensor(result,dtype=torch.bool,device=contexts.device)
