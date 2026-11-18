"""Official entry for REF0461: pinned vLLM ``swizzle_blockscale``.

Official source (commit ``f358f0578a83f4e1fe6fc49a492d3ae9362de8e5``):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/nvfp4_utils.py#L20-L56

The pinned Python function is imported from the frozen vLLM overlay and called
directly.  No argument mapping, layout adaptation or extension registration is
performed: the captured call sites pass a 2-D ``(M, K)`` ``torch.float8_e4m3fn``
CUDA tensor and receive a tensor with the same logical shape, exactly the
signature the baseline reference exposes.  The module import (and therefore any
vLLM import cost) happens once at import time and is outside the timed call.

Callable mapping (kept explicit because the two names differ):

    reference.py::v1_nvfp4_blockscale_swizzle  ->  official.py::compute_official
                                               ->  vllm...nvfp4_utils.swizzle_blockscale

Timing scope: :func:`compute_official` is the whole timed region, i.e. the
official Python entry plus the CUDA work it enqueues.  Import, case loading and
input restoration are outside the timed region.
"""
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale as _official_swizzle_blockscale,
)


def compute_official(scale):
    """Call the pinned official vLLM entry with the captured arguments."""
    return _official_swizzle_blockscale(scale)


# Signature-parity alias: the baseline reference name for the same operator.
v1_nvfp4_blockscale_swizzle = compute_official


def entry_identity():
    """Describe the exact loaded official symbol (used by the replay report)."""
    code = _official_swizzle_blockscale.__code__
    return {
        "symbol": "vllm.model_executor.layers.quantization.utils.nvfp4_utils.swizzle_blockscale",
        "file": code.co_filename,
        "first_lineno": code.co_firstlineno,
    }
