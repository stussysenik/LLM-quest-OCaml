"""Export GPT-2 past_key_values for the OCaml cache cross-check.

Greedy-decodes 2 tokens past the prompt (4 ids total), runs one forward
with use_cache=True, and saves per-layer K/V (batch squeezed to
(12, T, 64)) plus the input ids:

  - kv.input_ids   (I64, 1D)
  - pkv.k.{0..11}  (F32, 3D heads-major)
  - pkv.v.{0..11}  (F32, 3D heads-major)

Usage:
    python ocaml/parity/export_kv.py /data/tmp/ocaml-parity/gpt2_kv_ref.safetensors

Pinned stack: see parity/requirements.txt + parity/pinned.py.
"""

import sys

import torch
from safetensors.torch import save_file
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import pinned

pinned.check_pinned()


def main(out_path, prompt="Hello world", extra=2):
    model = pinned.align_parameters(
        GPT2LMHeadModel.from_pretrained("gpt2", revision=pinned.GPT2_REVISION).eval()
    )
    tok = GPT2Tokenizer.from_pretrained("gpt2", revision=pinned.GPT2_REVISION)
    ids = tok(prompt, return_tensors="pt").input_ids
    print(pinned.banner())
    with torch.no_grad():
        for _ in range(extra):
            out = model(ids)
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=-1)
        print(f"ids: {ids[0].tolist()}")
        out = model(ids, use_cache=True)
    tensors = {"kv.input_ids": ids[0].cpu().clone()}
    # transformers >= 5 returns a DynamicCache: per-layer .keys/.values
    for li, layer in enumerate(out.past_key_values.layers):
        tensors[f"pkv.k.{li}"] = layer.keys[0].cpu().clone().contiguous()
        tensors[f"pkv.v.{li}"] = layer.values[0].cpu().clone().contiguous()
    save_file(tensors, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/data/tmp/ocaml-parity/gpt2_kv_ref.safetensors")
