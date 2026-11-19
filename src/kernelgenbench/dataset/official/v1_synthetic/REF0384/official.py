# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/device_communicators/base_device_communicator.py#L269-L292
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

run = DeviceCommunicatorBase.all_gather
