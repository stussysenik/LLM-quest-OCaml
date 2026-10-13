"""Export HF weights + reference activations for the OCaml parity harness.

One script, one `--family` switch. The file layout, the convention probe
and the tensor naming are shared code; only the model class, tokenizer
and revision differ, and those live in the FAMILIES table below. Adding
a third family is a new export script plus one table entry — not a new
copy of the harness.

Writes ONE safetensors file containing:
  - every tensor of the model's state_dict() (cloned: tied weights like
    lm_head/wte share storage and save_file rejects that)
  - ref.input_ids      (I64, 1D squeezed)
  - ref.hidden.{0..L}  (F32, 2D squeezed: embeddings + block outputs +
    the post-final-norm state, batch dim removed)
  - ref.logits         (F32, 2D squeezed: seq x vocab)

Usage:
    python ocaml/parity/export_reference.py --family gpt2  /data/tmp/ocaml-parity/gpt2_ref.safetensors
    python ocaml/parity/export_reference.py --family llama /data/tmp/ocaml-parity/llama_ref.safetensors

Pinned stack: see parity/requirements.txt + parity/pinned.py. Re-export
a family's file together with the kv/gen references whenever the pin moves.
"""

import argparse

import torch
from safetensors.torch import save_file
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer

import pinned

pinned.check_pinned()

LLAMA_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
HEAD_KEY = "lm_head.weight"


def load_gpt2(prompt):
    model = GPT2LMHeadModel.from_pretrained("gpt2", revision=pinned.GPT2_REVISION).eval()
    tok = GPT2Tokenizer.from_pretrained("gpt2", revision=pinned.GPT2_REVISION)
    return model, tok(prompt, return_tensors="pt").input_ids


def load_llama(prompt):
    model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL, revision=pinned.LLAMA_REVISION).eval()
    tok = AutoTokenizer.from_pretrained(LLAMA_MODEL, revision=pinned.LLAMA_REVISION)
    return model, tok(prompt, return_tensors="pt").input_ids


FAMILIES = {"gpt2": load_gpt2, "llama": load_llama}


def final_norm_module(model):
    """The model's final norm. GPT-2 names it `ln_f`, Llama `norm` — and
    they are different modules (LayerNorm vs RMSNorm), so the parity
    stage name and the OCaml kernel both differ by family."""
    base = model.base_model
    for name in ("norm", "ln_f"):
        if hasattr(base, name):
            return getattr(base, name)
    raise RuntimeError(f"no final norm found on {type(base).__name__}")


def probe_convention(model, hidden_states, logits):
    """Which hidden state feeds the LM head, found by pairing instead of
    assumed.

    transformers 5.x appends the POST-final-norm state as the last entry
    for gpt2; that is a convention, not a contract, and Llama's final
    norm is RMSNorm rather than LayerNorm. Assuming it and being wrong
    reads as a numerical breach at parity time. So: the post-norm state
    is the one that reproduces the logits exactly, and the raw last-block
    state does not. Verify it here, print it, and fail loudly if the
    harness's slot assumption (ref.hidden.{L} is post-norm) no longer
    holds."""
    head = model.get_output_embeddings().weight.detach()
    with torch.no_grad():
        errors = [(float((h[0] @ head.T - logits[0]).abs().max()), i) for i, h in enumerate(hidden_states)]
    best_err, best = min(errors)
    layers = model.config.num_hidden_layers
    print(f"final norm: {type(final_norm_module(model)).__name__}")
    print(f"hidden states: {len(hidden_states)}, seq={logits.shape[1]}, layers={layers}")
    print(f"hidden[0] shape: {tuple(hidden_states[0].shape)}   hidden[-1] shape: {tuple(hidden_states[-1].shape)}")
    for err, i in sorted(errors, key=lambda e: e[1])[-3:]:
        print(f"  hidden[{i}] @ lm_head.T vs logits: max_abs_err={err:.3e}")
    print(f"post-final-norm state: hidden_states[{best}] (pairing error {best_err:.3e})")
    if best != layers or len(hidden_states) != layers + 1:
        raise RuntimeError(
            f"hidden-state convention changed: post-norm state is index {best} of "
            f"{len(hidden_states)}, harness expects index {layers} of {layers + 1}. "
            f"Fix export_reference.py's layout before re-exporting."
        )
    return best


def export(model, ids, out_path, drop_head=False):
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    layers = model.config.num_hidden_layers
    post_norm = probe_convention(model, out.hidden_states, out.logits)

    tensors = {k: v.cpu().clone().contiguous() for k, v in model.state_dict().items()}
    tensors["ref.input_ids"] = ids[0].cpu().clone()
    for i in range(layers):
        tensors[f"ref.hidden.{i}"] = out.hidden_states[i][0].cpu().clone().contiguous()
    tensors[f"ref.hidden.{layers}"] = out.hidden_states[post_norm][0].cpu().clone().contiguous()
    tensors["ref.logits"] = out.logits[0].cpu().clone().contiguous()

    # Tied embeddings: the head shares storage with embed_tokens/wte, so
    # state_dict() carries both keys and the file has both too. The loader
    # reads the config and asserts this — it must not silently fabricate a
    # head that is absent, or present-but-stale for an untied model.
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    print(f"tie_word_embeddings={tied}  {HEAD_KEY} in file={HEAD_KEY in tensors}")
    if not tied and HEAD_KEY not in tensors:
        raise RuntimeError(f"untied model with no {HEAD_KEY}: the loader would have to fabricate it")

    if drop_head:
        # What a genuinely tied checkpoint looks like on disk: HF's
        # save_pretrained drops shared tensors, so the head is simply
        # absent and the loader must alias the embedding matrix. Written
        # so gate C4 has a real tied input to assert against instead of
        # a hand-rolled fake.
        del tensors[HEAD_KEY]
        print(f"dropped {HEAD_KEY} — this file models tie_word_embeddings=true")

    save_file(tensors, out_path)
    print(f"saved {out_path} ({len(tensors)} tensors)")


def main(family, out_path, prompt="Hello world", drop_head=False):
    model, ids = FAMILIES[family](prompt)
    print(f"family: {family}  model: {model.config._name_or_path or model.config.model_type}")
    print(f"prompt {prompt!r} -> ids {ids[0].tolist()}")
    print(pinned.banner())
    export(model, ids, out_path, drop_head=drop_head)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=sorted(FAMILIES), default="gpt2")
    ap.add_argument("--drop-head", action="store_true", help="omit lm_head.weight (models a tied checkpoint)")
    ap.add_argument("out", nargs="?", help="output safetensors path")
    args = ap.parse_args()
    suffix = "_tied_ref" if args.drop_head else "_ref"
    default = f"/data/tmp/ocaml-parity/{args.family}{suffix}.safetensors"
    main(args.family, args.out or default, drop_head=args.drop_head)