"""Export GQA reference values for the OCaml step-D5 gate.

D5 is the last isolated Llama delta: Grouped-Query Attention, where K and
V are projected onto `num_kv_groups` heads instead of `num_heads` and each
K/V head serves `num_repeat` query heads.

Two references are written from the *same* weights and inputs, because
there are two things worth proving and one gate per thing:

- `d5.ref_hf`   — HF `LlamaAttention`. This is the implementation the
  ROADMAP names, and the one a real Llama checkpoint is gated against.
- `d5.ref_repo` — the repo's own `GroupedQueryAttention`
  (`llm_quest/gpt_to_llama3/llama_attention.py`), which is the Python
  this port actually mirrors. Matching HF is not automatically the same
  as matching the repo, and the port's contract is the repo.

The shapes are `Config.llama_tiny_test_config`: hidden 64, 4 heads,
2 kv heads, head_dim 16 — GQA on purpose. `tiny-random-LlamaForCausalLM`
has kv_heads == num_heads (4/4) and therefore *no* GQA at all, so a green
step E is not a green D5 (see ROADMAP.md).

Weights are randn / sqrt(fan_in) so activations stay O(1) and the dual
abs+rel gate is calibrated on magnitude rather than on the reference's own
fp32 rounding. W_q is deliberately a different *shape* from W_k/W_V
(num_heads * head_dim = 64 rows vs num_kv_groups * head_dim = 32, same
fan_in) so a transposed or mis-swapped projection cannot agree with the
reference by accident.

Writes (all F32, batch dim squeezed — attention is per-token over the
last dim, so [batch, seq, hidden] and [batch*seq, hidden] are the same
computation):
  d5.x          [batch*seq, d_in]
  d5.w_q        [num_heads    * head_dim, d_in]
  d5.w_k        [num_kv_groups * head_dim, d_in]
  d5.w_v        [num_kv_groups * head_dim, d_in]
  d5.w_o        [d_out, d_out]
  d5.b_o        [d_out]
  d5.ref_hf     [batch*seq, d_out]  — LlamaAttention(x)
  d5.ref_repo   [batch*seq, d_out]  — GroupedQueryAttention(x)

Usage:
    python ocaml/parity/export_gqa.py [out.safetensors]

Pinned stack: see parity/requirements.txt + parity/pinned.py.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

from llm_quest.common.rope import RoPE
from llm_quest.gpt_to_llama3.llama_attention import GroupedQueryAttention

import pinned

pinned.check_pinned()

SEED = 20250101
# BATCH is 1 on purpose. Unlike the FFN (norm/mlp exports), attention is
# NOT per-token: the causal mask spans a sequence, so squeezing the batch
# dim into the row dim would silently turn "sequence of 6, batch 2" into
# "sequence of 12" and apply the wrong mask. With batch=1 the squeeze is
# exact. Batching is covered by test_batched on the MHA path.
BATCH, SEQ = 1, 6
# Config.llama_tiny_test_config: hidden 64, 4 heads, 2 kv heads.
HIDDEN = 64
NUM_HEADS = 4
NUM_KV_GROUPS = 2
HEAD_DIM = HIDDEN // NUM_HEADS  # 16
D_OUT = NUM_HEADS * HEAD_DIM  # 64
ROPE_THETA = 10000.0


def main(out_path):
    print(f"module: {LlamaAttention.__module__}.{LlamaAttention.__name__}")
    print(f"repo:   {GroupedQueryAttention.__module__}.{GroupedQueryAttention.__name__}")
    print(
        f"shapes: hidden={HIDDEN} num_heads={NUM_HEADS} num_kv_groups={NUM_KV_GROUPS} "
        f"head_dim={HEAD_DIM} d_out={D_OUT} num_repeat={NUM_HEADS // NUM_KV_GROUPS}"
    )
    print(
        "convention: K/V projected onto num_kv_groups heads; each K/V head serves "
        "num_repeat CONTIGUOUS query heads (repeat_interleave, not tile); Q/K RoPE'd, V not"
    )
    print(f"inputs: seed={SEED} batch={BATCH} seq={SEQ}")
    print(pinned.banner())

    g = torch.Generator().manual_seed(SEED)

    def randn_mat(rows, cols):
        return torch.randn(rows, cols, generator=g) / math.sqrt(cols)

    x = torch.randn(BATCH, SEQ, HIDDEN, generator=g)

    w_q = randn_mat(NUM_HEADS * HEAD_DIM, HIDDEN)
    w_k = randn_mat(NUM_KV_GROUPS * HEAD_DIM, HIDDEN)
    w_v = randn_mat(NUM_KV_GROUPS * HEAD_DIM, HIDDEN)
    w_o = randn_mat(D_OUT, D_OUT)
    b_o = torch.randn(D_OUT, generator=g) / math.sqrt(D_OUT)

    # --- reference 1: HF LlamaAttention ---
    # attention_bias=True so that o_proj has a bias to copy (the repo's
    # GroupedQueryAttention builds out_proj with the nn.Linear default, i.e.
    # WITH bias). Q/K/V are then zeroed by hand, because Llama's Q/K/V are
    # bias-free (`bias=False` in the repo, `attention_bias=False` in real
    # Llama 3.2) and HF applies one flag to all four projections. The
    # o_proj bias is left real so the bias stays load-bearing in the gate.
    cfg = LlamaConfig(
        hidden_size=HIDDEN,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_GROUPS,
        max_position_embeddings=2048,
        rope_theta=ROPE_THETA,
        attention_bias=True,
        _attn_implementation="eager",
    )
    attn = LlamaAttention(cfg, layer_idx=0)
    with torch.no_grad():
        attn.q_proj.weight.copy_(w_q)
        attn.k_proj.weight.copy_(w_k)
        attn.v_proj.weight.copy_(w_v)
        attn.o_proj.weight.copy_(w_o)
        attn.o_proj.bias.copy_(b_o)
        # Llama's Q/K/V carry no bias.
        attn.q_proj.bias.zero_()
        attn.k_proj.bias.zero_()
        attn.v_proj.bias.zero_()
    assert attn.q_proj.bias.abs().max().item() == 0.0

    rot = LlamaRotaryEmbedding(config=cfg)
    position_embeddings = rot(x, torch.arange(SEQ).unsqueeze(0))

    # Additive causal mask: 0 where visible, finfo.min where masked.
    neg = torch.finfo(torch.float32).min
    causal = torch.triu(
        torch.full((BATCH, 1, SEQ, SEQ), neg, dtype=torch.float32), diagonal=1
    )

    with torch.no_grad():
        ref_hf, _ = attn(x, position_embeddings=position_embeddings, attention_mask=causal)

    # --- reference 2: the repo's own GroupedQueryAttention ---
    gqa = GroupedQueryAttention(
        d_in=HIDDEN, d_out=D_OUT, num_heads=NUM_HEADS, num_kv_groups=NUM_KV_GROUPS
    )
    with torch.no_grad():
        gqa.w_queries.weight.copy_(w_q)
        gqa.w_keys.weight.copy_(w_k)
        gqa.w_values.weight.copy_(w_v)
        gqa.out_proj.weight.copy_(w_o)
        gqa.out_proj.bias.copy_(b_o)

    # RoPE tables in the repo's own convention (ctx_len x head_dim).
    cos, sin = RoPE.compute_angles(base=ROPE_THETA, head_dim=HEAD_DIM, ctx_len=SEQ)
    # Boolean causal mask, True = masked (the repo's convention).
    bool_mask = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
    with torch.no_grad():
        ref_repo = gqa(x, bool_mask, cos, sin)

    hf_flat = ref_hf.reshape(BATCH * SEQ, D_OUT).cpu().clone().contiguous()
    repo_flat = ref_repo.reshape(BATCH * SEQ, D_OUT).cpu().clone().contiguous()
    hf_vs_repo = (ref_hf - ref_repo).abs().max().item()

    print(f"x{tuple(x.shape)} -> ref_hf{tuple(hf_flat.shape)}  |ref_hf|max={hf_flat.abs().max():.4f}")
    print(f"HF LlamaAttention vs repo GroupedQueryAttention: max_abs_err={hf_vs_repo:.3e}")
    if hf_vs_repo > 1e-5:
        raise RuntimeError(
            f"the two references disagree by {hf_vs_repo:.3e}; the export would be "
            f"gating against two different functions"
        )

    tensors = {
        "d5.x": x.reshape(BATCH * SEQ, HIDDEN).cpu().clone().contiguous(),
        "d5.w_q": w_q.cpu().clone().contiguous(),
        "d5.w_k": w_k.cpu().clone().contiguous(),
        "d5.w_v": w_v.cpu().clone().contiguous(),
        "d5.w_o": w_o.cpu().clone().contiguous(),
        "d5.b_o": b_o.cpu().clone().contiguous(),
        "d5.ref_hf": hf_flat,
        "d5.ref_repo": repo_flat,
    }

    save_file(tensors, out_path)
    print(f"saved {out_path} ({len(tensors)} tensors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="/tmp/ocaml-parity/gqa_ref.safetensors")
    main(ap.parse_args().out)