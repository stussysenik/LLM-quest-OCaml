"""Export SwiGLU reference values for the OCaml step-D2 gate.

Same contract as export_norm.py: the step-D rule is that each delta is
verified against PyTorch *before* anything is composed with it, so D2's
proof is one operator — the SwiGLU FFN — against the module HF actually
runs. The script writes the inputs (random x, random gate/up/down
weights) and LlamaMLP's output for them into one file, so the OCaml gate
cannot be comparing its own idea of the inputs against PyTorch's idea of
the inputs.

Two shapes, for the same reason D1 has two: at hidden=16 the fp32
reduction inside the projections has 16 terms, at hidden=256 it has 256,
and an accumulation-order mistake grows with the term count while
staying under the tolerance at 16. The tiny case is tiny-random-Llama's
own shape (16 -> 64).

The mid case is **square on purpose** (hidden == intermediate == 256):
that is the shape where a transposed down_proj multiplies to plausible
numbers instead of raising the inner-dim check — the silent-wrong
failure mode the [out, in] layout doctrine exists for. The non-square
tiny case is where the same mistake is loud.

Weights are randn / sqrt(fan_in) (Xavier scaling), not plain randn: the
gated product amplifies — plain randn weights put |ref| in the thousands,
where one fp32 ulp is already ~1e-3 absolute and a dual abs+rel gate
would be measuring the reference's rounding rather than our error. The
scaling keeps activations O(1) so the gate stays calibrated on
magnitude. x is plain randn, so silu sees its full range including the
negatives where it is not identity-like.

Also writes the silu extreme grid (d2.silu.x / d2.silu.ref): torch's
SiLU on fp32 at +/-100 and +/-800, where the naive sigmoid either
overflows (fp32: exp(100) = inf) or flushes to zero. -710 is
deliberately NOT in the torch grid: torch's fp32 SiLU flushes it to
-0.0 while the true value is ~-7e-306, so the fp64 kernel is pinned to
the *function* there by a separate gate, not to the fp32 reference.

Writes (all F32, batch dim squeezed — the FFN is per-token over the last
dim, so [batch, seq, hidden] and [batch*seq, hidden] are the same
computation and the squeeze only keeps the reader 2D):
  d2.<case>.x          [batch*seq, hidden]
  d2.<case>.gate_proj  [intermediate, hidden]
  d2.<case>.up_proj    [intermediate, hidden]
  d2.<case>.down_proj  [hidden, intermediate]
  d2.<case>.ref        [batch*seq, hidden]  — LlamaMLP(x)
  d2.silu.x            [9]  — the extreme grid
  d2.silu.ref          [9]  — torch.nn.SiLU on it

Usage:
    python ocaml/parity/export_mlp.py [out.safetensors]

Pinned stack: see parity/requirements.txt + parity/pinned.py.
"""

import argparse
import math

import torch
from safetensors.torch import save_file
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaMLP

import pinned

pinned.check_pinned()

SEED = 20250101
BATCH, SEQ = 2, 4
# (hidden, intermediate) per case. tiny is tiny-random-Llama's shape.
CASES = {"tiny": (16, 64, 4), "mid": (256, 256, 8)}
SILU_POINTS = [-800.0, -100.0, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0, 800.0]


def export_case(name, hidden, inter, heads, tensors):
    g = torch.Generator().manual_seed(SEED + hidden)

    x = torch.randn(BATCH, SEQ, hidden, generator=g)

    def randn_mat(rows, cols, fan_in):
        return torch.randn(rows, cols, generator=g) / math.sqrt(fan_in)

    gate = randn_mat(inter, hidden, hidden)
    up = randn_mat(inter, hidden, hidden)
    down = randn_mat(hidden, inter, inter)

    # The module under test, built the way LlamaForCausalLM builds it —
    # instantiated rather than hand-rolled, because a hand-rolled formula
    # in the reference would be the same formula this gate is verifying.
    # num_attention_heads is set only to satisfy LlamaConfig's own
    # hidden % heads validation; LlamaMLP never sees it.
    cfg = LlamaConfig(hidden_size=hidden, intermediate_size=inter, num_attention_heads=heads)
    mlp = LlamaMLP(cfg)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(gate)
        mlp.up_proj.weight.copy_(up)
        mlp.down_proj.weight.copy_(down)
        ref = mlp(x)

    # The same module in fp64, from the same fp32 weights widened (which is
    # lossless). LlamaMLP has no dtype pin of its own — every cast in it is
    # linear-algebra promotion — so unlike LlamaRMSNorm this is purely a
    # precision choice, and it is the OCaml port's: the port computes in
    # fp64, so the reference should too. Gating against an fp32 reference
    # would bound the gate by an error our kernel does not have. Measured
    # island (fp32 ref vs this) is printed for both cases.
    with torch.no_grad():
        mlp64 = LlamaMLP(cfg).double()
        mlp64.load_state_dict(mlp.state_dict())
        ref64 = mlp64(x.double())

    x2 = x.reshape(BATCH * SEQ, hidden)
    y = ref.reshape(BATCH * SEQ, hidden).cpu().clone().contiguous()
    y64 = ref64.reshape(BATCH * SEQ, hidden).cpu().clone().contiguous()
    island = (y.double() - y64).abs().max().item()
    print(
        f"{name}: x{tuple(x.shape)} -> {tuple(y.shape)}  gate{gate.shape} up{up.shape} "
        f"down{down.shape}  |ref|max={y.abs().max():.4f}  |x|max={x.abs().max():.4f}  "
        f"fp32-island={island:.3e}"
    )

    tensors[f"d2.{name}.x"] = x2.cpu().clone().contiguous()
    tensors[f"d2.{name}.gate_proj"] = gate.cpu().clone().contiguous()
    tensors[f"d2.{name}.up_proj"] = up.cpu().clone().contiguous()
    tensors[f"d2.{name}.down_proj"] = down.cpu().clone().contiguous()
    tensors[f"d2.{name}.ref"] = y
    tensors[f"d2.{name}.ref64"] = y64


def main(out_path):
    # The conventions this reference encodes, printed so that a change in
    # HF's module is visible at export time instead of arriving later as a
    # numerical breach with no explanation.
    print(f"module: {LlamaMLP.__module__}.{LlamaMLP.__name__}")
    print("convention: down_proj(silu(gate_proj(x)) * up_proj(x)), no biases, "
          "fp32 accumulation, weights [out, in]")
    print(f"inputs: seed={SEED} batch={BATCH} seq={SEQ} (batch squeezed; the FFN is per-token)")
    print(pinned.banner())

    tensors = {}
    for name, (hidden, inter, heads) in CASES.items():
        export_case(name, hidden, inter, heads, tensors)

    silu_x = torch.tensor(SILU_POINTS)
    silu_ref = torch.nn.SiLU()(silu_x)
    silu_ref64 = torch.nn.SiLU()(silu_x.double())
    print(
        "silu grid: " + "  ".join(
            f"{v:g}->{r:.3e}" for v, r in zip(SILU_POINTS, silu_ref.tolist())
        )
    )
    tensors["d2.silu.x"] = silu_x.cpu().clone().contiguous()
    tensors["d2.silu.ref"] = silu_ref.cpu().clone().contiguous()
    tensors["d2.silu.ref64"] = silu_ref64.cpu().clone().contiguous()

    save_file(tensors, out_path)
    print(f"saved {out_path} ({len(tensors)} tensors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="/data/tmp/ocaml-parity/mlp_ref.safetensors")
    main(ap.parse_args().out)
