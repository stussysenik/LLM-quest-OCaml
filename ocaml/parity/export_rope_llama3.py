"""Export RoPE llama3 scaling reference values for the OCaml step-D4 gate.

The step-D rule is that each delta is verified against PyTorch *before*
anything is composed with it. D4 verifies the llama3 frequency scaling on
the inverse-frequency table alone — before any attention uses it.

Uses [_compute_llama3_parameters] from transformers' modeling_rope_utils.py
as the reference implementation. The scaling adjusts inverse frequencies by
wavelength:
  - wavelen > low_freq_wavelen: divide by factor
  - wavelen < high_freq_wavelen: no change
  - otherwise: smooth interpolation

Writes (all F32):
  d4.inv_freq        [head_dim // 2]  — scaled inverse frequencies
  d4.base_inv_freq   [head_dim // 2]  — unscaled (for comparison)
  d4.cos             [ctx_len, head_dim]
  d4.sin             [ctx_len, head_dim]

Usage:
    python ocaml/parity/export_rope_llama3.py [out.safetensors]

Pinned stack: see parity/requirements.txt + parity/pinned.py.
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from transformers.modeling_rope_utils import _compute_llama3_parameters
from transformers.models.llama.configuration_llama import LlamaConfig

import pinned

pinned.check_pinned()

HEAD_DIM = 16
CTX_LEN = 32
BASE = 10000.0
# llama3 defaults from the original implementation
FACTOR = 8.0
LOW_FREQ_FACTOR = 1.0
HIGH_FREQ_FACTOR = 4.0
OLD_CTX_LEN = 8192


def main(out_path):
    print(f"module: {_compute_llama3_parameters.__module__}")
    print(
        "convention: llama3 scaling — wavelen > low_freq_wavelen: inv_freq / factor; "
        "wavelen < high_freq_wavelen: unchanged; otherwise: smooth interpolation"
    )
    print(
        f"inputs: head_dim={HEAD_DIM} ctx_len={CTX_LEN} base={BASE:g} "
        f"factor={FACTOR} low_freq_factor={LOW_FREQ_FACTOR} "
        f"high_freq_factor={HIGH_FREQ_FACTOR} old_ctx_len={OLD_CTX_LEN}"
    )
    print(pinned.banner())

    cfg = LlamaConfig(
        hidden_size=64,
        num_attention_heads=4,
        rope_theta=BASE,
        rope_scaling={
            "rope_type": "llama3",
            "factor": FACTOR,
            "low_freq_factor": LOW_FREQ_FACTOR,
            "high_freq_factor": HIGH_FREQ_FACTOR,
            "original_max_position_embeddings": OLD_CTX_LEN,
        },
    )

    inv_freq, attention_factor = _compute_llama3_parameters(cfg, seq_len=CTX_LEN)
    inv_freq = inv_freq.cpu().clone().contiguous()

    # Build cos/sin tables from the scaled inv_freq
    half = HEAD_DIM // 2
    positions = torch.arange(CTX_LEN, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq.float())
    angles = torch.cat([angles, angles], dim=-1)  # duplicate to head_dim
    cos = torch.cos(angles).cpu().clone().contiguous()
    sin = torch.sin(angles).cpu().clone().contiguous()

    # Unscaled base inv_freq for comparison
    base_inv_freq = 1.0 / (BASE ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))

    print(f"inv_freq: {inv_freq.shape}  |inv_freq|max={inv_freq.abs().max():.6e}")
    print(f"base_inv_freq: {base_inv_freq.shape}  |base_inv_freq|max={base_inv_freq.abs().max():.6e}")
    print(f"cos: {cos.shape}  sin: {sin.shape}  attention_factor={attention_factor}")

    tensors = {
        "d4.inv_freq": inv_freq,
        "d4.base_inv_freq": base_inv_freq.cpu().clone().contiguous(),
        "d4.cos": cos,
        "d4.sin": sin,
    }

    save_file(tensors, out_path)
    print(f"saved {out_path} ({len(tensors)} tensors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="/tmp/ocaml-parity/rope_llama3_ref.safetensors")
    main(ap.parse_args().out)
