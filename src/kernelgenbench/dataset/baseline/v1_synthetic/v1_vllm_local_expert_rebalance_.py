"""vLLM EPLB local expert-weight rebalance, one-rank equivalent.

Official source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/eplb/rebalance_execute.py#L305-L568
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/eplb/rebalance_execute.py#L501-L577
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/eplb/rebalance_execute.py#L669-L769

Contract: ``old_indices`` and ``new_indices`` are [num_layers, num_physical]
global physical -> logical expert maps.  For each layer, every weight tensor in
``expert_weights[layer]`` is updated in place so physical slot ``dst`` holds the
weight of logical expert ``new_indices[layer, dst]``; ``-1`` means an empty
slot and is skipped.  ``expert_buffer`` provides one staging tensor per weight
and is written for locally/remotely moved slots.  ``is_profile=True`` performs
no copies.  The upstream function is a distributed entry; this reference keeps
its one-rank local-copy branch and the same staging order.  The two-rank
all-gather/communicator branches are not represented here.
"""
import torch
import torch.distributed as dist


def v1_vllm_local_expert_rebalance_(old_indices, new_indices, expert_weights, expert_buffer, is_profile, ep_size=1, ep_rank=0, group=None):
    if is_profile:
        return
    for layer, (old, new) in enumerate(zip(old_indices.tolist(), new_indices.tolist())):
        n = expert_weights[layer][0].shape[0]
        local_old = old[ep_rank * n:(ep_rank + 1) * n]
        local_new = new[ep_rank * n:(ep_rank + 1) * n]
        snapshots = []
        for w in expert_weights[layer]:
            gathered = [torch.empty_like(w) for _ in range(ep_size)]
            if ep_size > 1:
                dist.all_gather(gathered, w, group=group)
            else:
                gathered[0].copy_(w)
            snapshots.append(gathered)
        local_first = {}; global_first = {}
        for i, e in enumerate(old):
            if e != -1 and e not in global_first:
                global_first[e] = i
        for i, e in enumerate(local_old):
            if e != -1 and e not in local_first:
                local_first[e] = i
        remote_primary = {}; staged = []
        for dst, e in enumerate(local_new):
            if e == -1 or e == local_old[dst]:
                continue
            if e in local_first:
                source = ep_rank * n + local_first[e]
            elif e in remote_primary:
                continue
            else:
                remote_primary[e] = dst
                source = global_first[e]
            for gathered, buf in zip(snapshots, expert_buffer):
                buf[dst].copy_(gathered[source // n][source % n])
            staged.append(dst)
        for dst in staged:
            for w, buf in zip(expert_weights[layer], expert_buffer):
                w[dst].copy_(buf[dst])
        for dst, e in enumerate(local_new):
            if e != -1 and e != local_old[dst] and e not in local_first and remote_primary.get(e) != dst:
                for w in expert_weights[layer]:
                    w[dst].copy_(w[remote_primary[e]])
