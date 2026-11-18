"""SGLang masks CPU-owned expert IDs in place before GPU dispatch.

Official source (pinned commit):
https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/srt/layers/moe/kt_ep_wrapper.py#L99-L116

Contract: ``topk_ids`` is an integer tensor of shape [tokens, top_k] in any
layout.  Every element ``>= num_gpu_experts`` is set to -1 in place; smaller
values (including already-negative sentinels) are unchanged.  The function
returns the same tensor object it received.  The upstream entry is wrapped in
``torch.compile(dynamic=True)``; this reference keeps the eager assignment that
the compiled function performs.
"""


def v1_mask_cpu_expert_ids(topk_ids, num_gpu_experts):
    topk_ids[topk_ids >= num_gpu_experts] = -1
    return topk_ids
