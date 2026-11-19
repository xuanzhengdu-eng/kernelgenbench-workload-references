# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/penalties.py#L107-L217
from vllm.v1.worker.gpu.sample.penalties import apply_penalties as v1_fused_sampling_penalties_

__all__ = ["v1_fused_sampling_penalties_"]
