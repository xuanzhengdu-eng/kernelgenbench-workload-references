"""DCP MLA extend: gather prefix KV then append new K-nope / K-rope.
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py#L100-L129
"""
import torch,torch.distributed as dist
def v1_sgl_dcp_extend_kv_gather_(cache_nope,cache_pe,local_indices,prefix_lens,k_nope,k_pe,dcp_kv_buffer,group):
    rank=dist.get_rank(group);world=dist.get_world_size(group);total=sum(prefix_lens);dn=cache_nope.shape[-1]
    if total:
        maxlen=max(prefix_lens);shape=(len(prefix_lens),maxlen)+tuple(cache_nope.shape[1:-1])+(cache_nope.shape[-1]+cache_pe.shape[-1],)
        merged=torch.zeros(shape,dtype=cache_nope.dtype,device=cache_nope.device);offset=0
        local=torch.cat((cache_nope[local_indices.long()],cache_pe[local_indices.long()]),dim=-1)
        for r,n in enumerate(prefix_lens):
            count=(n+world-1-rank)//world
            merged[r,rank:n:world]=local[offset:offset+count];offset+=count
        parts=[torch.empty_like(merged) for _ in range(world)];dist.all_gather(parts,merged,group=group)
        stacked=torch.stack(parts);out=[]
        for r,n in enumerate(prefix_lens):
            pos=torch.arange(n,device=cache_nope.device);out.append(stacked[pos%world,r,pos])
        prefix=torch.cat(out,dim=0);dcp_kv_buffer[:total].copy_(prefix)
    ext=dcp_kv_buffer.shape[0]-total
    dcp_kv_buffer[total:,...,:dn].copy_(k_nope[:ext]);dcp_kv_buffer[total:,...,dn:].copy_(k_pe[:ext])
    return None
