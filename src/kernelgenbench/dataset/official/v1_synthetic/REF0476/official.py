"""Official fixed-version entry for REF0476 (vLLM NVFP4 Marlin scale conversion).

Reference under test
--------------------
``reference.py`` (byte-identical to the KGB baseline
``v1_vllm_nvfp4_marlin_scales.py``) implements the same conversion as the
pinned vLLM function:

    upstream commit : f358f0578a83f4e1fe6fc49a492d3ae9362de8e5
    file            : vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py
    helper          : _nvfp4_compute_scale_factor  (L36-L55)
    entry           : nvfp4_marlin_process_scales  (L58-L122)
    real call site  : prepare_fp4_layer_for_marlin (L238-L242)

The reference and the official callable have the *same* positional signature
``(marlin_scales, scale_factor=None, a_dtype=None)`` and the same two-tuple
return ``(scales, scale_factor)``.  This module therefore re-exports the
upstream function object itself as ``run``; there is no argument reordering,
dtype conversion, layout coercion, wrapper, or extra kernel launch.  The timed
region is exactly the upstream Python entry and the PyTorch/CUDA kernels it
launches.  Everything else in this file is provenance reporting performed at
import time, outside the timed region.

Upstream branch note
--------------------
The upstream entry begins with ``if not (marlin_scales >= 0).all():`` which
performs a device-to-host reduction/synchronisation.  The reference omits that
warning-only check.  For the captured non-negative scales the check cannot
change the returned bytes; it is nevertheless inside the official timed region
and is therefore visible in the measured p50 values.  No branch is substituted
or bypassed here.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

PIN_ROOT = Path("/data/dxz/kgb-pinned")
PINNED_SRC = PIN_ROOT / "vllm/src/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5"
MAIN_SITE = PIN_ROOT / "vllm/main-site"
FRAMEWORK_COMMIT = "f358f0578a83f4e1fe6fc49a492d3ae9362de8e5"
EXPECTED_MODULE_SHA256 = (
    "78c19d495276c8376ebdc1a2639ef221ab1753cb0b89d6a00ac2446ffbc89b9e"
)
ENTRY_MODULE = "vllm.model_executor.layers.quantization.utils.marlin_utils_fp4"
ENTRY_ATTR = "nvfp4_marlin_process_scales"

_mod = importlib.import_module(ENTRY_MODULE)
run = getattr(_mod, ENTRY_ATTR)
compute_official = run

REFERENCE_SIGNATURE = ("marlin_scales", "scale_factor", "a_dtype")
OFFICIAL_SIGNATURE = (
    "marlin_scales",
    "scale_factor",
    "a_dtype",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_loaded_path = Path(_mod.__file__).resolve()
_loaded_sha256 = _sha256(_loaded_path)
if _loaded_sha256 != EXPECTED_MODULE_SHA256:
    raise RuntimeError(
        "REF0476 official module hash mismatch: "
        f"{_loaded_path} -> {_loaded_sha256}, expected {EXPECTED_MODULE_SHA256}"
    )
if run is not getattr(_mod, ENTRY_ATTR):
    raise RuntimeError("REF0476 official entry is not the module attribute")


def provenance() -> dict:
    """Describe the code that will actually execute (read-only, not timed)."""
    import platform

    import torch

    import vllm

    info = {
        "official_module": ENTRY_MODULE,
        "official_entry": f"{ENTRY_MODULE}.{ENTRY_ATTR}",
        "official_callable": repr(run),
        "loaded_from": str(_loaded_path),
        "loaded_sha256": _loaded_sha256,
        "pinned_source_tree": str(PINNED_SRC),
        "pinned_source_file": str(
            PINNED_SRC
            / "vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py"
        ),
        "pinned_source_sha256": _sha256(
            PINNED_SRC
            / "vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py"
        ),
        "framework_commit": FRAMEWORK_COMMIT,
        "vllm_version": getattr(vllm, "__version__", None),
        "vllm_package": str(Path(vllm.__file__).resolve().parent),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "adapter_overhead": "none: run is the upstream callable object",
        "signature": {
            "reference": REFERENCE_SIGNATURE,
            "official": OFFICIAL_SIGNATURE,
            "identical": REFERENCE_SIGNATURE == OFFICIAL_SIGNATURE,
        },
    }
    try:
        import triton

        info["triton_version"] = triton.__version__
    except Exception as exc:  # pragma: no cover - dependency report only
        info["triton_version"] = f"unavailable: {exc!r}"
    try:
        import safetensors

        info["safetensors_version"] = safetensors.__version__
    except Exception as exc:  # pragma: no cover - dependency report only
        info["safetensors_version"] = f"unavailable: {exc!r}"
    return info


if __name__ == "__main__":  # bare provenance dump, no GPU work
    print(json.dumps(provenance(), indent=1))
