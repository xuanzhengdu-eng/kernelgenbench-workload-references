# Source scope: vLLM Ming V3 ASR opt-in NoRepeatNGramLogitsProcessor reads request history and masks next tokens matching the current n-gram prefix. Local expresses the full-history rule with CPU loops and explicit slots/positions; source uses staged request state and tensorized masking, and is not a generic sampler hook.
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/models/ming_v3_asr_logits_processor.py#L10-L76
"""Ming ASR full-history n-gram masking of logits in place."""
import torch

def v1_vllm_no_repeat_ngram(logits,history,request_slots,positions,ngram_sizes):
    """Return same logits; prohibit historical next tokens for repeated prefix."""
    assert logits.ndim==2 and logits.is_floating_point() and logits.shape[1]>0
    rows,vocab=logits.shape;assert history.ndim==2 and history.shape[0]>0 and history.shape[1]>0
    assert history.dtype in (torch.int32,torch.int64) and request_slots.shape==positions.shape==(rows,)
    assert request_slots.dtype in (torch.int32,torch.int64) and positions.dtype in (torch.int32,torch.int64)
    assert ngram_sizes.shape==(history.shape[0],) and ngram_sizes.dtype in (torch.int32,torch.int64)
    assert all(t.device==logits.device for t in (history,request_slots,positions,ngram_sizes))
    assert ((request_slots>=0)&(request_slots<history.shape[0])).all() and ((positions>=-1)&(positions<history.shape[1])).all()
    assert ((ngram_sizes>=0)&(ngram_sizes.to(torch.int64)<2**31)).all() and torch.isfinite(logits).all()
    slots=request_slots.long().cpu().tolist();lengths=(positions+1).cpu().tolist();sizes=ngram_sizes.cpu().tolist();hist=history.cpu().tolist()
    for row,(slot,length) in enumerate(zip(slots,lengths)):
        n=sizes[slot]
        if n==0 or n>history.shape[1] or n>length:continue
        sequence=hist[slot][:length];tail=sequence[length-(n-1):] if n>1 else []
        blocked=set()
        for p in range(length-n+1):
            if n==1 or sequence[p:p+n-1]==tail:
                token=sequence[p+n-1]
                if 0<=token<vocab:blocked.add(token)
        if blocked:logits[row,torch.tensor(sorted(blocked),device=logits.device,dtype=torch.long)]=-float('inf')
    return logits
