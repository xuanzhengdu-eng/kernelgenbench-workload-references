"""LUT-B tile quantization with snapped Lloyd codebooks.

Source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/quantization/utils/lut_b_utils.py#L56-L200

Each ``8 x 64`` tile is scaled by a power of two chosen from its amax, fitted
with three initializations (farthest, quantile, uniform) and Lloyd iterations,
then packed as 192 bytes of 3-bit indices plus an eight-entry E4M3 codebook
and one UE8M0 byte.  Both ``num_iterations`` and ``max_tiles_per_chunk`` are
keyword-only, matching the upstream signature.
"""
import torch


@torch.no_grad()
def v1_vllm_quantize_lut_b(weight, *, num_iterations=16, max_tiles_per_chunk=4096):
    squeeze = weight.ndim == 2
    if squeeze:
        weight = weight.unsqueeze(0)
    if weight.ndim != 3:
        raise ValueError("LUT-B expects 2D or 3D weight")
    e, n, k = weight.shape
    if n % 8 or k % 64:
        raise ValueError("LUT-B requires N multiple of 8, K multiple of 64")
    nt, kt = n // 8, k // 64
    tiles = weight.reshape(e * nt, 8, k)
    packed = weight.new_empty((e * nt, kt, 192), dtype=torch.uint8)
    books = weight.new_empty((e * nt, kt, 8), dtype=torch.float8_e4m3fn)
    scales = weight.new_empty((e * nt, kt), dtype=torch.uint8)
    rows_per_chunk = max(1, max_tiles_per_chunk // kt)

    for start in range(0, e * nt, rows_per_chunk):
        stop = min(start + rows_per_chunk, e * nt)
        x = (
            tiles[start:stop]
            .float()
            .reshape(stop - start, 8, kt, 64)
            .transpose(1, 2)
            .reshape(-1, 512)
        )
        amax = x.abs().amax(1).clamp_min(torch.finfo(torch.float32).tiny)
        exponent = torch.ceil(torch.log2(amax / 448)).clamp(-127, 127)
        x = x * torch.exp2(-exponent)[:, None]
        preserve_zero = (x == 0).any(1)

        first = x.new_empty((x.shape[0], 8))
        first[:, 0] = x.mean(1)
        dist = (x - first[:, 0, None]).square()
        for i in range(1, 8):
            first[:, i] = x.gather(1, dist.argmax(1, keepdim=True)).squeeze(1)
            dist = torch.minimum(dist, (x - first[:, i, None]).square())

        ranks = ((torch.arange(8, device=x.device) + 0.5) * 64).long()
        quantile = x.sort(1).values[:, ranks]
        lo = x.amin(1, keepdim=True)
        hi = x.amax(1, keepdim=True)
        uniform = lo + (hi - lo) * torch.linspace(0, 1, 8, device=x.device)

        best_loss = x.new_full((x.shape[0],), float("inf"))
        best_codes = torch.empty_like(x, dtype=torch.uint8)
        best_centers = x.new_empty((x.shape[0], 8))

        for initial in (first, quantile, uniform):
            def normalize(centers):
                centers = centers.clamp(-448, 448).to(torch.float8_e4m3fn).float()
                zero_index = centers.abs().argmin(1)
                rows = preserve_zero.nonzero(as_tuple=True)[0]
                centers[rows, zero_index[rows]] = 0
                return centers.sort(1).values

            centers = normalize(initial)
            for _ in range(num_iterations):
                boundaries = ((centers[:, :-1] + centers[:, 1:]) * 0.5).contiguous()
                ids = torch.searchsorted(boundaries, x.contiguous())
                sums = torch.zeros_like(centers).scatter_add_(1, ids, x)
                counts = torch.zeros_like(centers).scatter_add_(
                    1, ids, torch.ones_like(x)
                )
                centers = normalize(
                    torch.where(counts > 0, sums / counts.clamp_min(1), centers)
                )
            boundaries = ((centers[:, :-1] + centers[:, 1:]) * 0.5).contiguous()
            ids = torch.searchsorted(boundaries, x.contiguous())
            loss = (centers.gather(1, ids) - x).square().sum(1)
            better = loss < best_loss
            best_loss = torch.where(better, loss, best_loss)
            best_codes[better] = ids[better].to(torch.uint8)
            best_centers[better] = centers[better]

        groups = best_codes.reshape(-1, 64, 8).int()
        words = torch.zeros(groups.shape[:-1], device=x.device, dtype=torch.int32)
        for i in range(8):
            words |= groups[..., i] << (3 * i)
        payload = torch.stack(
            (words & 255, (words >> 8) & 255, (words >> 16) & 255), -1
        ).to(torch.uint8).reshape(stop - start, kt, 192)
        packed[start:stop] = payload
        books[start:stop] = best_centers.to(torch.float8_e4m3fn).reshape(
            stop - start, kt, 8
        )
        scales[start:stop] = (exponent + 127).to(torch.uint8).reshape(
            stop - start, kt
        )

    packed = packed.reshape(e, nt, kt, 192)
    books = books.reshape(e, nt, kt, 8)
    scales = scales.reshape(e, nt, kt)
    if squeeze:
        return packed[0], books[0], scales[0]
    return packed, books, scales
