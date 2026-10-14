"""Export RoPE reference values for the OCaml step-D3 gate.

The step-D rule is that each delta is verified against PyTorch *before*
anything is composed with it, so D3's proof is one operator — RoPE —
against the implementation the repo actually runs. This script writes
the cos/sin tables, random Q/K inputs, and RoPE.apply's output for them
into one file, so the OCaml gate cannot be comparing its own idea of the
inputs against PyTorch's idea of the inputs.

Three kinds of reference are written, one per D3 gate:

1. **Full-forward apply** (d3.q_ref / d3.k_ref): RoPE.apply on the first
   seq_len positions — the prefill case, where positions are [0..seq-1].

2. **Cached-at-pos-p** (d3.q_cached_ref / d3.k_cached_ref): RoPE.apply
   on the last token alone, with position_ids=[pos] — the KV-cache decode
   case. The gate checks this equals row `pos` of the full-forward ref.

3. **Direct rotation** is not a separate tensor: the gate implements the
   explicit 2D rotation matrix in the test itself and compares it against
   the library's `apply`, so a bug in rotate_half or the cos/sin indexing
   cannot pass by agreeing with a reference that has the same bug.

The repo's RoPE operates on (b, n_head, seq, head_dim); the OCaml code
uses per-head (seq, head_dim) matrices. The export squeezes to per-head
(batch=1, n_head=1) — RoPE is applied per-head independently, so the
squeeze is exact, not an approximation.

Writes (all F32):
  d3.cos           [ctx_len, head_dim]
  d3.sin           [ctx_len, head_dim]
  d3.q             [seq_len, head_dim]
  d3.k             [seq_len, head_dim]
  d3.q_ref         [seq_len, head_dim]  — RoPE.apply(q, cos, sin)
  d3.k_ref         [seq_len, head_dim]  — RoPE.apply(k, cos, sin)
  d3.q_cached      [1, head_dim]        — q at position pos
  d3.k_cached      [1, head_dim]        — k at position pos
  d3.q_cached_ref  [1, head_dim]        — RoPE.apply(q_cached, cos, sin, position_ids=[pos])
  d3.k_cached_ref  [1, head_dim]        — RoPE.apply(k_cached, cos, sin, position_ids=[pos])
  d3.pos           [1]                  — the cached position

Usage:
    python ocaml/parity/export_rope.py [out.safetensors]

Pinned stack: see parity/requirements.txt + parity/pinned.py.
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_quest.common.rope import RoPE

import pinned

pinned.check_pinned()

SEED = 20250101
HEAD_DIM = 8
CTX_LEN = 16
SEQ_LEN = 8
BASE = 10000.0


def main(out_path):
    print(f"module: {RoPE.__module__}.{RoPE.__name__}")
    print(
        "convention: theta_i = base^(-2i/head_dim), angles duplicated end-to-end "
        "(halves, not interleaved), rotate_half + cos*x + sin*rotate_half(x)"
    )
    print(f"inputs: seed={SEED} head_dim={HEAD_DIM} ctx_len={CTX_LEN} seq_len={SEQ_LEN} base={BASE:g}")
    print(pinned.banner())

    g = torch.Generator().manual_seed(SEED)

    # Random Q, K — per-head (seq_len, head_dim)
    q = torch.randn(SEQ_LEN, HEAD_DIM, generator=g)
    k = torch.randn(SEQ_LEN, HEAD_DIM, generator=g)

    # cos/sin tables from the repo's own RoPE.compute_angles
    cos, sin = RoPE.compute_angles(base=BASE, head_dim=HEAD_DIM, ctx_len=CTX_LEN)

    # Full-forward reference: positions 0..seq_len-1 (prefill case)
    # RoPE.apply expects (b, n_head, seq, head_dim); squeeze back to (seq, head_dim)
    q_4d = q.unsqueeze(0).unsqueeze(0)
    k_4d = k.unsqueeze(0).unsqueeze(0)
    q_ref = RoPE.apply(q_4d, cos, sin).squeeze(0).squeeze(0).cpu().clone().contiguous()
    k_ref = RoPE.apply(k_4d, cos, sin).squeeze(0).squeeze(0).cpu().clone().contiguous()

    # Cached case: last token at an arbitrary position (decode case)
    pos = CTX_LEN - 1
    pos_ids = torch.tensor([[pos]])
    q_cached = q[-1:]
    k_cached = k[-1:]
    q_cached_ref = (
        RoPE.apply(q_cached.unsqueeze(0).unsqueeze(0), cos, sin, position_ids=pos_ids)
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .clone()
        .contiguous()
    )
    k_cached_ref = (
        RoPE.apply(k_cached.unsqueeze(0).unsqueeze(0), cos, sin, position_ids=pos_ids)
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .clone()
        .contiguous()
    )

    print(f"full-forward: q{tuple(q.shape)} -> q_ref{tuple(q_ref.shape)}  |q_ref|max={q_ref.abs().max():.4f}")
    print(f"cached: pos={pos}  q_cached_ref{tuple(q_cached_ref.shape)}  |q_cached_ref|max={q_cached_ref.abs().max():.4f}")

    tensors = {
        "d3.cos": cos.cpu().clone().contiguous(),
        "d3.sin": sin.cpu().clone().contiguous(),
        "d3.q": q.cpu().clone().contiguous(),
        "d3.k": k.cpu().clone().contiguous(),
        "d3.q_ref": q_ref,
        "d3.k_ref": k_ref,
        "d3.q_cached": q_cached.cpu().clone().contiguous(),
        "d3.k_cached": k_cached.cpu().clone().contiguous(),
        "d3.q_cached_ref": q_cached_ref,
        "d3.k_cached_ref": k_cached_ref,
        "d3.pos": torch.tensor([pos]),
    }

    save_file(tensors, out_path)
    print(f"saved {out_path} ({len(tensors)} tensors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="/tmp/ocaml-parity/rope_ref.safetensors")
    main(ap.parse_args().out)
