"""vLLM modular MoE runner: routed experts plus a separately returned shared expert.

Source (pinned commit):
https://github.com/vllm-project/vllm/blob/f358f0578a83f4e1fe6fc49a492d3ae9362de8e5/vllm/model_executor/layers/fused_moe/runner/moe_runner.py#L867-L937

``MoERunner._apply_quant_method`` runs the shared expert and the routed experts
and returns ``(shared_output, fused_output)`` as two separate tensors.  This
reference reproduces that single-rank, unquantized FP32 arithmetic:

* routing: ``topk`` over the router logits, then either a softmax over the
  selected logits (``renormalize=True``) or ``exp(value - logsumexp(logits))``;
* routed experts: ``silu(gate) * up`` then the down projection, scaled by the
  token's routing weight and accumulated per expert;
* shared expert: the same gated MLP applied to ``shared_x``, returned on its own.

Shapes: ``x[T,D]``, ``logits[T,E]``, ``shared_x[T,D]``, routed gate/up
``w13[E,2F,D]``, routed down ``w2[E,D,F]``, shared gate/up ``shared_w13[2G,D]``
and shared down ``shared_w2[D,G]``.  No bias, no EP/TP, no quantization and no
deferred finalize.  The SGLang dual-stream branch (merged single output) is a
different return structure and is tracked separately, as are the locally
invented ``sglang_graph``/``sglang_v4`` conventions, which have no upstream
counterpart and are not part of this reference.
"""
import torch
import torch.nn.functional as F


def v1_vllm_routed_shared(x, logits, shared_x, w13, w2, shared_w13, shared_w2,
                          topk, renormalize):
    t, d = x.shape
    e = logits.shape[1]
    f = w2.shape[-1]
    g = shared_w2.shape[-1]
    if (logits.shape != (t, e) or shared_x.shape != (t, d)
            or w13.shape != (e, 2 * f, d) or w2.shape != (e, d, f)
            or shared_w13.shape != (2 * g, d) or shared_w2.shape != (d, g)
            or not 0 < topk <= e):
        raise ValueError("invalid shapes")
    val, ids = logits.topk(topk, dim=-1, sorted=False)
    probs = (val.softmax(-1) if renormalize
             else (val - torch.logsumexp(logits, -1, keepdim=True)).exp())
    sg, su = F.linear(shared_x, shared_w13).chunk(2, -1)
    shared = F.linear(F.silu(sg) * su, shared_w2)
    routed = torch.zeros_like(x)
    for expert in range(e):
        for j in range(topk):
            rows = torch.nonzero(ids[:, j] == expert).flatten()
            gate, up = F.linear(x[rows], w13[expert]).chunk(2, -1)
            y = F.linear(F.silu(gate) * up, w2[expert]) * probs[rows, j, None]
            routed.index_add_(0, rows, y)
    return shared, routed
