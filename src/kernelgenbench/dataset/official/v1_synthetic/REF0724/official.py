"""Fixed-version official entry for REF0724 (vLLM v1 greedy argmax).

Upstream source (pinned):
  https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/v1/worker/gpu/sample/greedy_argmax.py
  ``greedy_argmax``     L38-L66
  ``_argmax_partial``   L8-L26
  ``_argmax_finish``    L29-L35

The captured real entry is
``vllm.v1.worker.gpu.sample.greedy_argmax.greedy_argmax``.  This module imports
that symbol unchanged and exposes it as ``run``; ``v1_greedy_argmax`` is an
alias so the reference's callable signature maps 1:1.  No argument, dtype,
stride, storage or branch adaptation is performed: every frozen case already
supplies a CUDA 2-D bfloat16 tensor with unit vocabulary stride and vocabulary
151936, i.e. the split-Triton branch of the official function.

Timing scope: everything inside ``greedy_argmax`` (the branch predicate, the
``out`` allocation, ``values``/``indices`` scratch allocation and the two Triton
kernel launches) is inside the measured region.  ``run`` adds exactly one
Python call frame.  Nothing else is wrapped, converted or copied here.
"""
import hashlib
import importlib

ENTRY = "vllm.v1.worker.gpu.sample.greedy_argmax.greedy_argmax"
MODULE = "vllm.v1.worker.gpu.sample.greedy_argmax"
# sha256 of vllm/v1/worker/gpu/sample/greedy_argmax.py at the pinned commit; the
# installed overlay file is byte-identical to `git show <rev>:<path>`.
EXPECTED_SOURCE_SHA256 = (
    "6c2897be330b1de9a22257b192120d23bd69f84c085acee4ea346e4f98b778a5"
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load():
    module = importlib.import_module(MODULE)
    entry = getattr(module, "greedy_argmax")
    path = module.__file__
    observed = _sha256(path)
    if observed != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"pinned official source mismatch: {path} sha256={observed} "
            f"expected={EXPECTED_SOURCE_SHA256}"
        )
    if getattr(entry, "__module__", None) != MODULE:
        raise RuntimeError(f"official symbol resolved outside {MODULE}: {entry!r}")
    return module, entry


_module, greedy_argmax = _load()


def official_source():
    """Provenance of the loaded official implementation (import-time facts)."""
    path = _module.__file__
    observed = _sha256(path)
    return {
        "module": MODULE,
        "entry": ENTRY,
        "file": path,
        "sha256": observed,
        "expected_sha256": EXPECTED_SOURCE_SHA256,
        "pinned": observed == EXPECTED_SOURCE_SHA256,
        "symbol_module": greedy_argmax.__module__,
    }


def run(logits):
    """Call the pinned official entry; equivalent to the reference signature."""
    return greedy_argmax(logits)


v1_greedy_argmax = run
