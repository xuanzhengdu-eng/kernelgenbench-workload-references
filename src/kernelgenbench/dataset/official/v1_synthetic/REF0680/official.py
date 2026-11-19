# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/kvcache/kvcache.py#L48-L53
# https://github.com/sgl-project/sglang/blob/62384de1ae5e3bbd3159afd1639525b0bf39fbf9/python/sglang/kernels/ops/kvcache/kvcache.py#L69-L120
from sglang.kernels.ops.kvcache.kvcache import can_use_store_cache as can_use
from sglang.kernels.ops.kvcache.kvcache import store_cache as run

__all__ = ["run", "can_use"]
