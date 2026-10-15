"""Export RMSNorm reference values for the OCaml step-D1 gate.

The step-D rule is that each delta is verified against PyTorch *before*
anything is composed with it, so D1's proof is one operator — RMSNorm —
against the module HF actually runs. This script writes the inputs
(random x, random weight) and HF's output for them into one file, so the
OCaml gate cannot be comparing its own idea of the inputs against PyTorch's
idea of the inputs.

Two shapes, because the tiny one hides what the mid-size catches: at
hidden=16 the fp32 reduction over the last dim has 16 terms, at hidden=2048
it has 2048, and an accumulation-order mistake grows with the term count
while staying under the tolerance at 16.

Writes (batch dim squeezed — RMSNorm is per-token over the last dim, so
[batch, seq, hidden] and [batch*seq, hidden] are the same computation and
the squeeze only keeps the reader 2D):
  d1.<case>.x      [batch*seq, hidden]  F32  — the input, as fed to both refs
  d1.<case>.w      [hidden]             F32
  d1.<case>.eps    [1]                  F32  — the tiny-random Llama config's eps
  d1.<case>.ref    [batch*seq, hidden]  F32  — LlamaRMSNorm(x, w), unpatched
  d1.<case>.ref64  [batch*seq, hidden]  F64  — torch.nn.functional.rms_norm

Two references, because LlamaRMSNorm.forward hard-casts:

    hidden_states = hidden_states.to(torch.float32)

so even handed a float64 input it computes in float32, and `.ref` carries an
fp32 island. Measured for the mid case, that island is 7.767e-07 absolute —
which was, to every printed digit, what gate D1a reported. The gate was
measuring HF's dtype policy, and its margin against a 1e-6 bound was 1.29x,
so it would have flipped green→red on a host whose BLAS reduces in a
different order. `.ref64` is torch's own kernel, which computes in the input
dtype: gating against it separates "our kernel is wrong" from "HF cast the
input", and lets D1a calibrate its bound from the island instead of from a
constant nobody re-derives. See STUDY.md §1.

Usage:
    python ocaml/parity/export_norm.py [out.safetensors]

Pinned stack: see parity/requirements.txt + parity/pinned.py. The eps is
carried in the file rather than hardcoded in the OCaml gate because it is
an *input* to the op, and the whole point of gate D1d is that the value
fed to `rms_norm` is `Config.norm_eps` of the config — a constant written
down twice is a constant that can disagree with itself.
"""

import argparse

import torch
from safetensors.torch import save_file
from transformers.models.llama.modeling_llama import LlamaRMSNorm

import pinned

pinned.check_pinned()

SEED = 20250101
BATCH, SEQ = 2, 4
# The two hidden sizes of the gate: 16 is the shape where a bug in the
# reduction order cannot show, 2048 is the shape where it does.
CASES = {"tiny": 16, "mid": 2048}


def export_case(name, hidden, tensors):
    eps = 1e-6
    g = torch.Generator().manual_seed(SEED + hidden)

    x = torch.randn(BATCH, SEQ, hidden, generator=g)
    w = torch.randn(hidden, generator=g)

    # The module under test, built the way LlamaForCausalLM builds it —
    # instantiated rather than hand-rolled, because a hand-rolled formula
    # in the reference would be the same formula this gate is verifying.
    norm = LlamaRMSNorm(hidden, eps=eps)
    with torch.no_grad():
        norm.weight.copy_(w)
        ref = norm(x)

    x2 = x.reshape(BATCH * SEQ, hidden)
    y = ref.reshape(BATCH * SEQ, hidden).cpu().clone().contiguous()

    # Torch's own kernel, in the input dtype — no cast to fp32 anywhere in
    # it. Same fp32 inputs widened to fp64, so the only difference from
    # `ref` is HF's cast and the arithmetic precision that follows from it.
    with torch.no_grad():
        y64 = (
            torch.nn.functional.rms_norm(x2.double(), (hidden,), w.double(), eps)
            .cpu()
            .clone()
            .contiguous()
        )
    island = (y.double() - y64).abs().max().item()
    print(
        f"{name}: x{tuple(x.shape)} -> {tuple(y.shape)}  w{hidden}  eps={eps:g}  "
        f"|ref|max={y.abs().max():.4f}  |x|max={x.abs().max():.4f}  "
        f"fp32-island={island:.3e}"
    )

    tensors[f"d1.{name}.x"] = x2.cpu().clone().contiguous()
    tensors[f"d1.{name}.w"] = w.cpu().clone().contiguous()
    tensors[f"d1.{name}.eps"] = torch.tensor([eps])
    tensors[f"d1.{name}.ref"] = y
    tensors[f"d1.{name}.ref64"] = y64


def main(out_path):
    # The conventions this reference encodes, printed so that a change in
    # HF's module is visible at export time instead of arriving later as a
    # numerical breach with no explanation.
    print(f"module: {LlamaRMSNorm.__module__}.{LlamaRMSNorm.__name__}")
    print("convention: y = w * x / sqrt(mean(x^2, dim=-1) + eps)  "
          "(no mean subtraction, no bias, eps inside the sqrt, fp32 accumulation)")
    print(f"inputs: seed={SEED} batch={BATCH} seq={SEQ} (batch squeezed; RMSNorm is per-token)")
    print(pinned.banner())

    tensors = {}
    for name, hidden in CASES.items():
        export_case(name, hidden, tensors)

    save_file(tensors, out_path)
    print(f"saved {out_path} ({len(tensors)} tensors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="/data/tmp/ocaml-parity/norm_ref.safetensors")
    main(ap.parse_args().out)