# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/device_communicators/all2all.py#L183-L198
# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/parallel_state.py#L812-L824
from vllm.distributed.parallel_state import GroupCoordinator


def combine_into_output(group, hidden_states, output, dim=0, sizes=None):
    return group.reduce_scatterv_into_output(
        hidden_states, output, dim=dim, sizes=sizes
    )
