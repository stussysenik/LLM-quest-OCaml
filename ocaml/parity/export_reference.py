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


def force_fp64_reference(model):
    """Neutralise HF's internal fp32 downcasts for the duration of the export.

    Casting the model to float64 is not enough on its own. `LlamaRMSNorm`
    hard-casts inside its own forward:

        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        ...

    so every norm is still computed in fp32 no matter what dtype the model
    is in, and the "float64 reference" carries an fp32 island. Measured on
    tiny-random-Llama, that island is worth 1.26e-09 at block_0, 1.87e-07
    at block_1 and 2.20e-08 at the logits — the same three numbers the
    OCaml gate reported before this was patched, which is how it was
    identified. HF's eager attention also pins its softmax to float32, but
    measuring that separately showed it contributes exactly 0.0 here.

    This is not a change to the model — it is the same arithmetic at the
    precision the OCaml port claims. It has to happen at export time,
    because the alternative is a gate calibrated to ~1e-7 that silently
    measures HF's dtype choices rather than our kernels.
    """
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    def fp64_forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * hidden_states * torch.rsqrt(variance + self.variance_epsilon)

    LlamaRMSNorm.forward = fp64_forward
    return model


def load_gpt2(prompt):
    model = pinned.align_parameters(
        GPT2LMHeadModel.from_pretrained("gpt2", revision=pinned.GPT2_REVISION).eval()
    )
    tok = GPT2Tokenizer.from_pretrained("gpt2", revision=pinned.GPT2_REVISION)
    return model, tok(prompt, return_tensors="pt").input_ids


def load_llama(prompt):
    model = pinned.align_parameters(
        AutoModelForCausalLM.from_pretrained(LLAMA_MODEL, revision=pinned.LLAMA_REVISION).eval()
    )
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
    # Weights stay float32 — they are exactly what the checkpoint holds, and
    # the OCaml loaders assert F32 shapes against it.
    tensors = {k: v.cpu().clone().contiguous() for k, v in model.state_dict().items()}
    tensors["ref.input_ids"] = ids[0].cpu().clone()

    # Reference activations are computed in float64. The gate's tolerance is
    # bounded by the *less precise* of the two implementations, and fp32 is
    # not the bottleneck-free side: GPT-2's ~3000-magnitude outlier dims
    # mean a 12-layer fp32 run disagrees with the same run in fp64 by ~2e-6
    # relative. Gating a faithful fp64 OCaml port at 1e-6 against that would
    # measure PyTorch's rounding (and shift with the host's BLAS), so the
    # reference is produced at the precision we are actually claiming.
    # Both sides then sum in fp64 from bit-identical inputs; what remains is
    # summation order, ~1e-14 relative.
    model = model.double()
    model = force_fp64_reference(model)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)

    layers = model.config.num_hidden_layers
    post_norm = probe_convention(model, out.hidden_states, out.logits)

    for i in range(layers):
        tensors[f"ref.hidden.{i}"] = out.hidden_states[i][0].cpu().clone().contiguous()
    tensors[f"ref.hidden.{layers}"] = out.hidden_states[post_norm][0].cpu().clone().contiguous()
    tensors["ref.logits"] = out.logits[0].cpu().clone().contiguous()
    print(
        "reference dtypes: weights F32 (as checkpointed), activations "
        f"{out.logits.dtype} — see safetensors.ml mat_of_f64"
    )

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