# Source scope: vLLM pinned Punica LoRA wrapper allocates per-adapter alignment workspaces and calls moe_lora_align_block_size; the C++ small-batch expert kernel performs adapter/expert bucketing and block padding. Local only models count<1024, explicit active adapters, and block-padding outputs with CPU token loops; it does not reproduce CUDA launch or all EP/mapping cases.
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/lora/punica_wrapper/punica_gpu.py#L360-L449
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/csrc/moe/moe_align_sum_kernels.cu#L634-L675
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L2497-L2526
"""vLLM small-batch MoE-LoRA routing and block alignment."""
import torch

def v1_vllm_moe_lora_align(topk_ids,token_lora_mapping,num_experts,block_size,max_loras,max_num_tokens_padded,max_num_m_blocks,sorted_token_ids,experts_ids,num_tokens_post_pad,adapter_enabled,lora_ids,expert_map=None):
    """Mutate output workspaces in place; skipped adapter slots are unchanged."""
    assert topk_ids.ndim==2 and topk_ids.dtype==torch.int32 and topk_ids.is_contiguous()
    t,topk=topk_ids.shape;count=t*topk
    assert t>0 and topk>0 and count<1024 and 0<num_experts<=64 and block_size>0 and max_loras>0
    assert token_lora_mapping.shape==(t,) and token_lora_mapping.dtype==torch.int32
    assert lora_ids.shape==(max_loras+1,) and adapter_enabled.shape==(max_loras,)
    assert sorted_token_ids.shape==(max_loras*max_num_tokens_padded,) and experts_ids.shape==(max_loras*max_num_m_blocks,) and num_tokens_post_pad.shape==(max_loras,)
    assert all(x.dtype==torch.int32 and x.is_contiguous() and x.device==topk_ids.device for x in (token_lora_mapping,lora_ids,adapter_enabled,sorted_token_ids,experts_ids,num_tokens_post_pad))
    assert max_num_tokens_padded>=count+num_experts*(block_size-1) and max_num_m_blocks>=(max_num_tokens_padded+block_size-1)//block_size
    if expert_map is not None:assert expert_map.dtype==torch.int32 and expert_map.shape==(num_experts,) and expert_map.device==topk_ids.device
    ids=topk_ids.cpu().reshape(-1).tolist();mapping=token_lora_mapping.cpu().tolist();active=lora_ids.cpu().tolist();enabled=adapter_enabled.cpu().tolist();emap=None if expert_map is None else expert_map.cpu().tolist()
    selected=[i for i in active if 0<=i<max_loras and enabled[i]]
    assert len(set(selected))==len(selected) and all(i>=-1 for i in active)
    for adapter in selected:
        groups=[[] for _ in range(num_experts)]
        for i,expert in enumerate(ids):
            if mapping[i//topk]!=adapter or not 0<=expert<num_experts:continue
            local=expert if emap is None else emap[expert]
            if local<0:continue
            assert local<num_experts
            groups[local].append(i)
        tokens=[];blocks=[]
        for expert,values in enumerate(groups):
            padded=((len(values)+block_size-1)//block_size)*block_size
            tokens.extend(values+[count]*(padded-len(values)));blocks.extend([expert]*(padded//block_size))
        assert len(tokens)<=max_num_tokens_padded
        tokens.extend([count]*(max_num_tokens_padded-len(tokens)))
        blocks.extend([-1]*(max_num_m_blocks-len(blocks)))
        ss=adapter*max_num_tokens_padded;es=adapter*max_num_m_blocks
        sorted_token_ids[ss:ss+max_num_tokens_padded]=torch.tensor(tokens,dtype=torch.int32,device=topk_ids.device)
        experts_ids[es:es+max_num_m_blocks]=torch.tensor(blocks,dtype=torch.int32,device=topk_ids.device)
        num_tokens_post_pad[adapter]=sum(((len(g)+block_size-1)//block_size)*block_size for g in groups)
    return None
