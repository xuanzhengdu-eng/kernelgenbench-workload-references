"""Official fixed-version entry for REF0375 (vLLM DSA indexer Q RoPE + FP8 quant).

The PyTorch baseline under test is
``kernelgenbench.dataset.baseline.v1_synthetic.v1_vllm_indexer_q_rope_quant``
(byte-identical copy in ``reference.py`` next to this file).  The official
implementation captured for this reference is the pinned vLLM Triton path:

    upstream commit : f358f0578a83f4e1fe6fc49a492d3ae9362de8e5
    file            : vllm/model_executor/layers/sparse_attn_indexer.py
    kernel          : _fused_indexer_q_rope_quant_kernel  (L191-L272, @triton.jit)
    python wrapper  : fused_indexer_q_rope_quant          (L274-L315)
    call site       : vllm/model_executor/models/deepseek_v2.py L797
                      (Indexer.forward, `use_fused_indexer_q` branch L786)

The reference and the official wrapper take the *same* seven positional
arguments in the same order, with the same dtypes and the same return tuple, so
this module performs no argument reordering, no dtype conversion, no layout
coercion and no extra kernel launch.  ``run`` is therefore the official symbol
itself (re-exported, not re-wrapped), which keeps the timed region exactly the
upstream entry: Triton launch + the two ``torch.empty_like`` output allocations
performed inside ``fused_indexer_q_rope_quant``.  Everything else in this file
is provenance reporting, not part of the timed path.

Timing scope
------------
``run`` == ``fused_indexer_q_rope_quant``: both output allocations, the Triton
grid launch and the kernel execution are inside the window.  Nothing outside
the entry is added by this adapter, so adapter overhead is zero by
construction (verified in replay_check.py by comparing ``run``'s identity with
``fused_indexer_q_rope_quant``).
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

# The reference signature and the official signature are identical; this alias
# exists only so the mapping is explicit and reviewable.
REFERENCE_SIGNATURE = (
    "positions", "q", "cos_sin_cache", "weights",
    "softmax_scale", "head_scale", "is_neox",
)

_ENTRY_MODULE = "vllm.model_executor.layers.sparse_attn_indexer"
_ENTRY_ATTR = "fused_indexer_q_rope_quant"

_mod = importlib.import_module(_ENTRY_MODULE)
run = getattr(_mod, _ENTRY_ATTR)
compute_official = run

# ``vllm.model_executor.models.deepseek_v2`` imports the very same object; that
# is the object the capture wrapper observed (see case.json "entry").
_deepseek_v2 = importlib.import_module("vllm.model_executor.models.deepseek_v2")
assert _deepseek_v2.fused_indexer_q_rope_quant is run, (
    "call-site symbol is not the same object as the captured entry"
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provenance():
    """Describe the code that will actually execute (read-only, not timed)."""
    vllm = importlib.import_module("vllm")
    kernel = getattr(_mod, "_fused_indexer_q_rope_quant_kernel", None)
    kernel_file = None
    if kernel is not None:
        fn = getattr(kernel, "fn", kernel)
        kernel_file = getattr(getattr(fn, "__code__", None), "co_filename", None)
    return {
        "official_module": _ENTRY_MODULE,
        "official_entry": f"{_ENTRY_MODULE}.{_ENTRY_ATTR}",
        "official_callable": repr(run),
        "loaded_from": str(Path(_mod.__file__).resolve()),
        "loaded_sha256": _sha256(_mod.__file__),
        "vllm_version": getattr(vllm, "__version__", None),
        "vllm_package": str(Path(vllm.__file__).resolve().parent),
        "triton_kernel": getattr(kernel, "__name__", None),
        "triton_kernel_file": kernel_file,
        "call_site_symbol_identity": True,
        "adapter_overhead": "none: run is the upstream callable object",
    }


if __name__ == "__main__":  # bare provenance dump, no GPU work
    print(json.dumps(provenance(), indent=1))
