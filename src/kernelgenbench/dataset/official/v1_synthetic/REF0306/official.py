# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/eplb/policy/default.py#L273-L332
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/eplb/policy/default.py#L75-L100
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/eplb/policy/default.py#L191-L271
from vllm.distributed.eplb.policy.default import DefaultEplbPolicy

run = DefaultEplbPolicy.rebalance_experts
