"""Official vLLM entry for REF0692 ``v1_vllm_hf3fs_scatter_kv_``.

Pinned upstream source
---------------------
* Entry ``scatter_kv_caches``:
  https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/gather_scatter_helper.py#L110-L164
* Triton kernel ``kv_cache_scatter_kernel``:
  https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/gather_scatter_helper.py#L10-L57
* Real call site (``operation != "gather"`` branch):
  https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py#L975-L1002
* One-time ``_kvcache_ptrs`` construction used as the adaptation model:
  https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/hf3fs_connector.py#L529-L572

The real connector registers a uniform ``[B, H, T, C]`` per-layer view, keeps
the pointer vector ``torch.tensor([cache.data_ptr() ...], dtype=int64,
device=cuda)`` from registration time, and on every load task passes that
vector plus ``_local_total_tokens``, the load buffer ``[L, H, N, C]``, the
``range(block * T, block * T + T)`` token list, ``T``, ``H``, ``C`` and
``first_cache.stride()`` to ``scatter_kv_caches``.

Contract implemented here
-------------------------
``v1_vllm_hf3fs_scatter_kv_(kv_caches, total_token_in_kvcache, src_tensor,
token_indices, tokens_per_block)`` has the same signature and side effects as
``reference.py``: mutate every layer cache in place, return ``None`` and never
write ``src_tensor``.  The only work done by this adapter is the same mapping
the connector does: derive ``H``/``C``/strides from the uniform cache view and
build the int64 CUDA pointer vector.  ``prepare()`` performs that mapping once
and ``run()`` then calls the pinned official Python entry directly.

Timing scope
------------
``run()`` is the timed entry: it performs argument forwarding only and calls
``scatter_kv_caches``, which launches ``kv_cache_scatter_kernel``.  The Triton
launch, the internal ``async_tensor_h2d(token_indices)`` and the GPU kernel are
inside the timed region.  ``prepare()`` (pointer tensor construction) is a
one-time registration cost in the real connector and is measured separately,
never folded into the per-call timings.  ``v1_vllm_hf3fs_scatter_kv_`` builds
the binding per call for direct substitution use; its per-call binding cost is
reported as ``binding_us`` in ``results.json``.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

OFFICIAL_MODULE = (
    "vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.gather_scatter_helper"
)
OFFICIAL_SYMBOL = "scatter_kv_caches"
EXPECTED_FRAMEWORK_COMMIT = "f358f0578a83f4e1fe6fc49a492d3ae9362de8e5"
EXPECTED_HELPER_SHA256 = (
    "f14496a946572db2fa79a1bcd499dc62a1922bcacfd75c50770fb7a9b8983699"
)

_official_fn = None
_official_info: dict | None = None


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_official() -> tuple[object, dict]:
    """Import the pinned official symbol and verify the loaded file bytes."""
    global _official_fn, _official_info
    if _official_fn is not None:
        return _official_fn, _official_info  # type: ignore[return-value]
    module = importlib.import_module(OFFICIAL_MODULE)
    fn = getattr(module, OFFICIAL_SYMBOL)
    path = inspect.getsourcefile(fn)
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"official source file not found for {fn!r}")
    digest = sha256_file(path)
    if digest != EXPECTED_HELPER_SHA256:
        raise RuntimeError(
            "official gather_scatter_helper.py hash mismatch: "
            f"{digest} != {EXPECTED_HELPER_SHA256}"
        )
    try:
        import triton

        triton_version = triton.__version__
    except Exception:  # pragma: no cover - only used for reporting
        triton_version = None
    try:
        import vllm

        vllm_version = vllm.__version__
    except Exception:  # pragma: no cover
        vllm_version = None
    _official_info = {
        "module": OFFICIAL_MODULE,
        "symbol": OFFICIAL_SYMBOL,
        "source_file": path,
        "source_sha256": digest,
        "expected_source_sha256": EXPECTED_HELPER_SHA256,
        "framework_commit": EXPECTED_FRAMEWORK_COMMIT,
        "vllm_version": vllm_version,
        "triton_version": triton_version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    _official_fn = fn
    return fn, _official_info


@dataclass(frozen=True)
class Binding:
    """The per-registration state the HF3FS connector builds once."""

    kv_caches_ptrs: torch.Tensor
    num_layers: int
    num_heads: int
    content_size: int
    tokens_per_block: int
    kv_cache_strides: tuple[int, int, int, int]
    device: torch.device


def prepare(kv_caches: Sequence[torch.Tensor], tokens_per_block: int | None = None) -> Binding:
    """Replicate ``_setup_kv_cache_config`` for an in-memory cache list.

    Mirrors upstream: require uniform 4-D ``[B, H, T, C]`` views, take strides
    from the first view, and build the int64 CUDA pointer vector once.
    """
    if not isinstance(kv_caches, (list, tuple)) or len(kv_caches) == 0:
        raise TypeError("kv_caches must be a non-empty list of tensors")
    first = kv_caches[0]
    if first.dim() != 4:
        raise ValueError(f"expected a 4-D [B, H, T, C] view, got {tuple(first.shape)}")
    blocks, heads, block_size, content = (int(x) for x in first.shape)
    strides = tuple(int(x) for x in first.stride())
    if tokens_per_block is not None and int(tokens_per_block) != block_size:
        raise ValueError(
            f"tokens_per_block={tokens_per_block} does not match cache T={block_size}"
        )
    for i, cache in enumerate(kv_caches):
        if tuple(cache.shape) != tuple(first.shape):
            raise ValueError(f"layer {i} shape {tuple(cache.shape)} != {tuple(first.shape)}")
        if tuple(cache.stride()) != strides:
            raise ValueError(f"layer {i} stride {tuple(cache.stride())} != {strides}")
        if cache.dtype != first.dtype:
            raise ValueError(f"layer {i} dtype {cache.dtype} != {first.dtype}")
        if cache.device != first.device:
            raise ValueError(f"layer {i} device {cache.device} != {first.device}")
    ptrs = torch.tensor(
        [int(cache.data_ptr()) for cache in kv_caches],
        dtype=torch.int64,
        device=first.device,
    )
    return Binding(
        kv_caches_ptrs=ptrs,
        num_layers=len(kv_caches),
        num_heads=heads,
        content_size=content,
        tokens_per_block=block_size,
        kv_cache_strides=strides,  # type: ignore[arg-type]
        device=first.device,
    )


def run(
    binding: Binding,
    total_token_in_kvcache: int,
    src_tensor: torch.Tensor,
    token_indices: Iterable[int],
) -> None:
    """Timed entry: forward to pinned ``scatter_kv_caches`` (launch + kernel)."""
    # The pinned entry accepts a list[int] directly; do not re-copy it per call.
    indices = token_indices if isinstance(token_indices, list) else [int(x) for x in token_indices]
    fn, _ = load_official()
    expected = (binding.num_layers, binding.num_heads, len(indices), binding.content_size)
    if tuple(src_tensor.shape) != expected:
        raise ValueError(f"src_tensor shape {tuple(src_tensor.shape)} != {expected}")
    if src_tensor.dtype != torch.bfloat16 and src_tensor.dtype not in (
        torch.float16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError(f"unsupported source dtype {src_tensor.dtype}")
    # The pinned kernel indexes source_ptr with row-major offsets and ignores
    # source strides, so upstream requires a contiguous [L, H, N, C] buffer.
    if not src_tensor.is_contiguous():
        raise ValueError("pinned scatter_kv_caches requires a contiguous src_tensor")
    return fn(
        binding.kv_caches_ptrs,
        int(total_token_in_kvcache),
        src_tensor,
        indices,
        binding.tokens_per_block,
        binding.num_heads,
        binding.content_size,
        binding.kv_cache_strides,
    )


def v1_vllm_hf3fs_scatter_kv_(
    kv_caches: Sequence[torch.Tensor],
    total_token_in_kvcache: int,
    src_tensor: torch.Tensor,
    token_indices: Iterable[int],
    tokens_per_block: int,
) -> None:
    """Reference-signature equivalent entry; binding is built per call.

    This is the drop-in replacement for ``reference.v1_vllm_hf3fs_scatter_kv_``.
    It does the same one-time mapping the connector does, but per invocation,
    so its measured cost includes the pointer-vector construction.  Use
    ``prepare()`` + ``run()`` for the connector-equivalent timing.
    """
    binding = prepare(kv_caches, tokens_per_block)
    return run(binding, total_token_in_kvcache, src_tensor, token_indices)


def runtime_info() -> dict:
    """Report the actually loaded runtime (call after CUDA is initialised)."""
    _, info = load_official()
    out = dict(info)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        uuid = None
        try:
            uuid = str(props.uuid)
        except Exception:
            uuid = None
        out["gpu"] = {
            "name": props.name,
            "uuid": uuid,
            "total_memory_bytes": int(props.total_memory),
            "physical_index_env": os.environ.get("LEASED_PHYSICAL_GPU"),
            "expected_uuid_env": os.environ.get("EXPECTED_GPU_UUID"),
        }
    out["hostname"] = os.uname().nodename
    out["python"] = sys.version.split()[0]
    out["image_id"] = "sha256:ce429874fcb2f22e87abec76226e454b90390720c14fc3e5238d0ec564008a27"
    return out


__all__ = [
    "Binding",
    "EXPECTED_FRAMEWORK_COMMIT",
    "EXPECTED_HELPER_SHA256",
    "OFFICIAL_MODULE",
    "OFFICIAL_SYMBOL",
    "load_official",
    "prepare",
    "run",
    "runtime_info",
    "v1_vllm_hf3fs_scatter_kv_",
]
