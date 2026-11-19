# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/device_communicators/pynccl.py#L429-L455
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

run = PyNcclCommunicator.reduce
