"""vLLM LoRA ID-to-slot and sampler/embedding index metadata.
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/lora/punica_wrapper/utils.py#L52-L148
"""
import torch

def v1_lora_index_mapping(index_mapping,prompt_mapping,lora_index_to_id,max_loras,vocab_size,device):
    table={v:i for i,v in enumerate(lora_index_to_id) if v is not None}
    base=[table[v] if v>0 else -1 for v in index_mapping]
    emb=[table[v] if v>0 else 0 for v in index_mapping]
    sampler=[table[v] if v>0 else -1 for v in prompt_mapping]
    indices=torch.tensor([list(index_mapping),base,emb],dtype=torch.long,device=device)
    sampler=torch.tensor(sampler,dtype=torch.long,device=device)
    base_out=indices[1];emb_out=indices[2]*vocab_size
    padded=torch.where(sampler==-1,max_loras-1,sampler)
    padded=torch.arange(len(sampler),device=device,dtype=torch.long)+padded*len(sampler)
    return base_out,sampler,padded,emb_out,[base_out.numel(),sampler.numel(),padded.numel(),emb_out.numel()]
