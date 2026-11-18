# Source scope: SGLang EAGLE tree selection first step repeats hidden states and builds root metadata; later steps expand parent scores, top-k select, gather IDs/hidden rows and form tree metadata. Local follows this structure, with torch.topk and explicit shape assumptions instead of fast_topk/compiled execution.
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/speculative/spec_utils.py#L481-L547
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/utils/common.py#L507-L513
import torch

def v1_sgl_select_tree_tokens(i,topk_p,topk_index,hidden_states,scores,topk):
    """SGLang EAGLE draft-tree token/hidden selection with first/later step."""
    if i==0:
        ids=topk_index.flatten();h=hidden_states.repeat_interleave(topk,dim=0) if hidden_states is not None else None
        parents=torch.arange(-1,topk,dtype=torch.long,device=ids.device).expand(topk_p.shape[0],-1)
        return ids,h,topk_p,(topk_p.unsqueeze(1),topk_index,parents)
    k2=topk*topk;expanded=scores.unsqueeze(2)*topk_p.view(-1,topk,topk)
    sorted_scores,selected=torch.max(expanded.flatten(1),dim=-1,keepdim=True) if topk==1 else torch.topk(expanded.flatten(1),topk,dim=-1)
    tree_ids=topk_index.view(-1,k2);ids=tree_ids.gather(1,selected).flatten()
    if hidden_states is not None and hidden_states.shape[0]>0:
        offsets=torch.arange(0,hidden_states.shape[0],step=topk,device=ids.device).repeat_interleave(topk)
        hidden_states=hidden_states[selected.flatten()//topk+offsets]
    return ids,hidden_states,sorted_scores,(expanded,tree_ids,selected+(k2*(i-1)+topk))
