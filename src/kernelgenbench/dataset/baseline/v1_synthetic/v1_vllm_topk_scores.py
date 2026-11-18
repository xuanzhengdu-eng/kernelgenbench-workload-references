"""vLLM sampled-token score + top-k + rank, no per-request overrides.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/logprob.py#L84-L108
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/logprob.py#L111-L185
Finite logits only, sampled token ID first then top-k or per-request custom IDs (duplicates retained).
Output score can be FP32 logits or logprobs; rank counts scores >= sample,
so all ties count. Optional cumulative boundaries pass through unchanged.
"""
from typing import NamedTuple
import torch
class LogprobsTensors(NamedTuple):
    logprob_token_ids:torch.Tensor
    logprobs:torch.Tensor
    selected_token_ranks:torch.Tensor
    cu_num_generated_tokens:list|None=None
    cu_num_generated_tokens_tensor:torch.Tensor|None=None

def v1_vllm_topk_scores(logits,num_logprobs,sampled_token_ids,cu_num_logits=None,logprob_token_ids_state=None,expanded_idx_mapping=None,max_per_req_token_ids=0,logits_mode=False):
    assert 0<=num_logprobs<=logits.shape[1]
    top=logits.topk(num_logprobs,dim=-1).indices if num_logprobs else None
    valid=None
    if max_per_req_token_ids:
        assert logprob_token_ids_state is not None and expanded_idx_mapping is not None
        ncols=max(num_logprobs,max_per_req_token_ids);ids=sampled_token_ids.new_zeros((logits.shape[0],ncols+1));valid=torch.zeros_like(ids,dtype=torch.bool);ids[:,0]=sampled_token_ids;valid[:,0]=True
        counts=logprob_token_ids_state.num_token_ids.gpu;custom=logprob_token_ids_state.token_ids.gpu
        for i,req in enumerate(expanded_idx_mapping.tolist()):
            n=int(counts[req]);chosen=custom[req,:n] if n else (top[i] if top is not None else ids.new_empty(0));ids[i,1:1+chosen.numel()]=chosen;valid[i,1:1+chosen.numel()]=True
    else:
        ids=sampled_token_ids.unsqueeze(-1)
        if num_logprobs:ids=torch.cat((ids,top),dim=1)
    vals=logits.float() if logits_mode else logits.float().log_softmax(dim=-1)
    scores=vals.gather(1,ids.long())
    if valid is not None:scores=scores.masked_fill(~valid,-torch.inf)
    selected=logits.gather(1,sampled_token_ids.long().unsqueeze(1));ranks=(logits>=selected).sum(1).to(torch.int64)
    return LogprobsTensors(ids,scores,ranks,None if isinstance(cu_num_logits,torch.Tensor) else cu_num_logits,cu_num_logits if isinstance(cu_num_logits,torch.Tensor) else None)
