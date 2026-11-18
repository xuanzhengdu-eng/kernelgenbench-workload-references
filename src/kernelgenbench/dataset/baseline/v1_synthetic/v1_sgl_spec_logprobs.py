"""SGLang speculative token logprobs with optional top-k and requested IDs.
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/logprob_processor.py#L386-L459
Writes fields on logits_output and returns None. Exactly one of accept_index
and chain_stride is specified; accept indices retain negative-index behavior.
The original-logprob environment flag is adapted to explicit original_logprob.
"""
import torch

def v1_sgl_spec_logprobs(batch,logits_output,predict,accept_index=None,chain_stride=None,original_logprob=False):
    assert (accept_index is None)!=(chain_stride is None)
    b=len(batch.seq_lens);logits=logits_output.next_token_logits
    if accept_index is not None:
        width=accept_index.shape[1];ind=accept_index.long().reshape(-1);selected=logits[ind];tokens=predict[ind]
    else:
        width=chain_stride;assert logits.shape[0]==b*width;selected=logits;tokens=predict
    if batch.sampling_info.is_all_greedy or original_logprob:lp=torch.log_softmax(selected,dim=-1)
    else:
        temperatures=torch.repeat_interleave(batch.sampling_info.temperatures,width,dim=0);lp=torch.log_softmax(selected/temperatures,dim=-1)
    lp=lp.clamp_min(torch.finfo(lp.dtype).min)
    logits_output.next_token_logprobs=lp.gather(1,tokens.long().reshape(-1,1)).reshape(b,width)
    if batch.top_logprobs_nums and any(k>0 for k in batch.top_logprobs_nums):
        ks=[k for k in batch.top_logprobs_nums for _ in range(width)];vals,ids=lp.topk(max(ks),dim=-1)
        logits_output.next_token_top_logprobs_val=[vals[i,:k] for i,k in enumerate(ks)]
        logits_output.next_token_top_logprobs_idx=[ids[i,:k] for i,k in enumerate(ks)]
    if batch.token_ids_logprobs and any(ids is not None for ids in batch.token_ids_logprobs):
        lists=[ids for ids in batch.token_ids_logprobs for _ in range(width)];vals=[];ids_out=[]
        for i,ids in enumerate(lists):
            ids=[] if ids is None else ids;index=torch.tensor(ids,dtype=torch.long,device=lp.device);vals.append(lp[i,index]);ids_out.append(ids)
        logits_output.next_token_token_ids_logprobs_val=vals;logits_output.next_token_token_ids_logprobs_idx=ids_out
    return None
