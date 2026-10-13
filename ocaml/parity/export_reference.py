"""Export HF gpt2 weights + reference activations for the OCaml parity harness.

Writes ONE safetensors file containing:
  - every tensor of GPT2LMHeadModel.state_dict() (cloned: tied weights
    like lm_head/wte share storage and save_file rejects that)
  - ref.input_ids      (I64, 1D squeezed)
  - ref.hidden.{0..12} (F32, 2D squeezed: embeddings + 12 block outputs,
    exactly HF output_hidden_states=True, batch dim removed)
  - ref.logits         (F32, 2D squeezed: seq x vocab)

Usage:
    python ocaml/parity/export_reference.py /data/tmp/ocaml-parity/gpt2_ref.safetensors

Pinned stack: see parity/requirements.txt + parity/pinned.py. Re-export
this file together with the kv/gen references whenever the pin moves.
"""

import sys

import torch
from safetensors.torch import save_file
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import pinned

pinned.check_pinned()


def main(out_path, prompt="Hello world"):
    model = GPT2LMHeadModel.from_pretrained("gpt2", revision=pinned.GPT2_REVISION).eval()
    print(f"activation: {model.config.activation_function}")
    tok = GPT2Tokenizer.from_pretrained("gpt2", revision=pinned.GPT2_REVISION)
    ids = tok(prompt, return_tensors="pt").input_ids
    print(f"prompt {prompt!r} -> ids {ids[0].tolist()}")
    print(pinned.banner())

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    tensors = {k: v.cpu().clone().contiguous() for k, v in model.state_dict().items()}
    tensors["ref.input_ids"] = ids[0].cpu().clone()
    for i, h in enumerate(out.hidden_states):
        tensors[f"ref.hidden.{i}"] = h[0].cpu().clone().contiguous()
    tensors["ref.logits"] = out.logits[0].cpu().clone().contiguous()

    print(f"hidden states: {len(out.hidden_states)}, seq={ids.shape[1]}")
    save_file(tensors, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/data/tmp/ocaml-parity/gpt2_ref.safetensors")
