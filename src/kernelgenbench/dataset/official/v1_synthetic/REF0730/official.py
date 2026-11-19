# https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/_custom_ops.py#L382-L395
import vllm._custom_ops as _custom_ops

v1_repetition_penalty_ = _custom_ops.apply_repetition_penalties
