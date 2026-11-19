# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/backends/mla/triton_mla.py#L246-L345
from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl

run = TritonMLAImpl.forward_mqa
