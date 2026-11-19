# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/attention/ops/ultraquant/triton_store.py#L107-L351
from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store

run = ultraquant_store
__all__ = ["run", "ultraquant_store"]
