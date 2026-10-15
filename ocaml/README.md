# LLM Quest — OCaml port

Pure-OCaml (stdlib only, no Owl/torch bindings) re-implementation of the
Python/PyTorch repo at the workspace root. Goal, per the project brief:
**SOTA of model understanding** — every op spelled out, zero magic.

Scope: `llm_quest/gpt`, `llm_quest/gpt_to_llama3`, and
`llm_quest/multimodal`. Twelve other packages in `llm_quest/` are not
ported — see `ROADMAP.md` § "Out of scope".

## Layout

| OCaml | Python counterpart |
|---|---|
| `lib/config.ml` | `config.py` (GPT family) |
| `lib/tensor.ml` | `torch` ops used across `llm_quest/` |
| `lib/norm.ml` | `LayerNorm`, `GELU` in `gpt_transformer_block.py`; `LlamaRMSNorm` |
| `lib/mlp.ml` | SwiGLU — `LlamaMLP` (transformers), `SiLU`/`FFN` in `llama_transformer_block.py` |
| `lib/rope.ml` | `RoPE` + `compute_angles_llama3` (transformers) |
| `lib/kv_cache.ml` | `KVCache` in `llm_quest/utils.py` |
| `lib/attention.ml` | `gpt_attention.py` (v1/v2/v3 + `MultiHeadAttention`), `GroupedQueryAttention` |
| `lib/block.ml` | `FFN`, `TransformerBlock` in `gpt_transformer_block.py` |
| `lib/gpt.ml` | `llm_quest/gpt/gpt_model.py` (`GPTModel`) |
| `lib/sampling.ml` | `sampling*` + `generate_simple_loop` in `llm_quest/generate.py` |
| `lib/loss.ml` | `cross_entropy` in `vlm_engine.py` / `vit_engine.py` |
| `lib/safetensors.ml` | HF safetensors format (shared by all four loaders) |
| `lib/gpt2_loader.ml` | HF `gpt2` naming map (`Conv1D`, `ln_f`, `lm_head`) |
| `lib/llama.ml` | Llama forward pass (RMSNorm + RoPE + GQA + SwiGLU) |
| `lib/llama_loader.ml` | HF `LlamaForCausalLM` naming map (`nn.Linear`, RMSNorm, SwiGLU) |
| `lib/vit.ml` | `llm_quest/multimodal/vision_transformer/vit_model.py` (`ViTModel`) |
| `lib/vlm.ml` | `ViTAdapter` + `vlm_engine.py`'s early fusion + `vlm_generation.py`'s loop |
| `lib/image.ml` | `F.resize` / `F.to_tensor` / `F.normalize` in `vlm_generation.py` |
| `lib/vit_loader.ml` | `ViTModel.state_dict()` — incl. the 4-D `conv_proj` |
| `lib/gpt_repo_loader.ml` | `llm_quest.gpt.GPTModel.state_dict()` — `emb_dict`, `trf_blocks.N.*` |
| `lib/vlm_loader.ml` | the fused checkpoint: `vit.` + `adapter.` + `gpt.` namespaces |
| `bin/main.ml` | PART A demo in `generate.py __main__` |

**Three weight-naming schemes coexist**, which is not redundancy but the
actual state of the Python:

| Loader | Reads | Produced by |
|---|---|---|
| `gpt2_loader.ml` | `transformer.wte.weight`, `transformer.h.N.*` | HF `gpt2` |
| `llama_loader.ml` | `model.embed_tokens.weight`, `model.layers.N.*` | HF `LlamaForCausalLM` |
| `gpt_repo_loader.ml` | `emb_dict.weight`, `trf_blocks.N.*` | `llm_quest.gpt.GPTModel` |

`vlm_training.py` builds the VLM's decoder from `GPTModel`, so the fused
checkpoint's `gpt.` half is in the third scheme.

## Build

**The Makefile is the build.** It invokes `ocamlopt` directly and is what
`make ci` uses; the `dune` files exist so an editor resolves modules, and
`dune test` runs nothing (the suites are plain executables, not `(test)`
stanzas).

```bash
cd ocaml
make            # unit gates + demo
make ci         # the whole thing, ~2 min
make demo-run
make clean
```

Verified on OCaml **4.14.1** with no dune installed: the full library and
all 14 suites compile warning-free. No `dune`-only features are used.

`dune build && dune exec ./bin/main.exe` also works where dune is present.

## Deliberate simplifications

- Dropout is identity at inference (all shipped configs use `0.0`).
- No tokenizer. Generation parity is gated on **token IDs**, not decoded
  text — the reference file carries the tokenized prompt, so both sides
  start from identical input. `Vlm.generate_cached` returns ids for the
  same reason.
- **No training.** No optimizers, LR schedules, AMP, gradient clipping or
  checkpoint writing. What came across is the *measurement* side
  (`lib/loss.ml`, `Vlm.vlm_loss`, `Vit.batch_loss`/`batch_accuracy`).
- **The image resize is not torchvision's.** `to_tensor` and ImageNet
  `normalize` are bit-identical to torch; `resize_bilinear` implements
  `interpolate(mode="bilinear", align_corners=False)`, not the
  antialiased integer-domain `F.resize` that `vlm_generation.py` calls.
  See `lib/image.ml`.
- `vlm_training.py` writes PyTorch `.pth` pickles; this port reads
  safetensors only and has no pickle reader.

Each is documented at its own definition in code as well as here.

## Design notes

`config.ml` carries both text families in one record: shared fields at the
top level, everything that differs behind `arch` (`Gpt2` / `Llama`).
Code that needs a shape reads an accessor — `Config.max_positions`,
`norm_eps`, `ffn_inner`, `kv_heads`, `head_dim`, `has_learned_positions`
— so shared code (block stack, attention, cache, loader) never matches on
family. Llama-only knobs (`rope_theta`, `rope_scaling`,
`tie_word_embeddings`) raise on a GPT-2 config rather than returning a
plausible-looking default.

Every linear weight in the project is `[out, in]`. The one exception is
`nn.Conv2d`, deliberately kept as `[emb, C*P*P]` kernels because
flattening it is arithmetically fine and silently destroys the gate that
would notice.

## Running everything

`make ci` is the one-command green-tree proof: **274 PASS across 14
suites**, plus 14 GPT-2 parity stages (embeddings, 11 blocks + final
norm, logits), a 12-layer KV cross-check and
the Llama layout/logits boundary check. About two minutes on one core.

```bash
cd ocaml && make ci
```

| Suite | Gates | What it anchors on |
|---|---|---|
| `test_all` | 42 | self-consistency |
| `test_batched` | 9 | batch ≡ single |
| `test_kv` | 9 | KV-cache invariants |
| `test_loader` | 16 | GPT-2/HF loader + `[out, in]` layout |
| `test_norm` | 15 | `LlamaRMSNorm`, and torch's fp64 kernel |
| `test_mlp` | 22 | `LlamaMLP`, fp32 and fp64 |
| `test_rope` | 13 | `RoPE` + `llama3` scaling |
| `test_gqa` | 18 | `LlamaAttention` + this repo's own |
| `test_vit` | 25 | `ViTModel` |
| `test_vlm` | 30 | adapter + early fusion |
| `test_loader_vlm` | 26 | the three multimodal loaders, preprocessing, caption loop |
| `test_image` | 17 | `to_tensor`, normalize, resize |
| `test_loss` | 27 | `ignore_index`, accuracy |
| `parity_gen` | 5 | greedy generation vs `generate()` |

### Generating the references

The PyTorch stages need reference files, generated **once** against the
pinned stack in `parity/requirements.txt` (~635 MB, none tracked in git).
torch goes in a separate step, from the CPU index only — the reason is in
`parity/requirements.txt`'s header and is worth reading before changing it.

```bash
python3 -m venv /data/tmp/venv
/data/tmp/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
/data/tmp/venv/bin/pip install -r parity/requirements.txt

PY=/data/tmp/venv/bin/python; D=/data/tmp/ocaml-parity
$PY parity/export_reference.py --family gpt2  $D/gpt2_ref.safetensors
$PY parity/export_reference.py --family llama $D/llama_ref.safetensors
$PY parity/export_reference.py --family llama --drop-head $D/llama_tied_ref.safetensors
$PY parity/export_kv.py       $D/gpt2_kv_ref.safetensors
$PY parity/export_gen.py      $D/gpt2_gen_ref.safetensors
$PY parity/export_norm.py     $D/norm_ref.safetensors
$PY parity/export_mlp.py      $D/mlp_ref.safetensors
$PY parity/export_rope.py     $D/rope_ref.safetensors
$PY parity/export_rope_llama3.py $D/rope_llama3_ref.safetensors
$PY parity/export_gqa.py      $D/gqa_ref.safetensors
$PY parity/export_vit.py      $D/vit_ref.safetensors
$PY parity/export_vlm.py      $D/vlm_ref.safetensors
$PY parity/export_loader.py   $D/vlm_loader_ref.safetensors
$PY parity/export_image.py    $D/image_ref.safetensors
$PY parity/export_loss.py     $D/loss_ref.safetensors
```

If `/data` is not writable on your machine, put them elsewhere and point
the Makefile at it: `make ci PARITY_DIR=/tmp/ocaml-parity`.

### Why the references look the way they do

Two conventions that are easy to get backwards, both load-bearing:

- **Weights are F32, reference activations are F64.** A checkpoint is
  float32 because that is what `torch.save` writes. Activations are
  float64 because a gate compares two implementations of one function
  and its tolerance is set by the less precise of the two. Against an
  fp32 reference the gates breached at ~1e-6 — because *PyTorch disagreed
  with its own fp64 run by 2.3e-6*. Moving the activations to fp64 took
  every stage to between 0.0 (the embedding lookup) and 2.3e-12, and the
  tolerance to **1e-9**.
- **The inputs are stored next to the outputs.** So a gate cannot pass by
  comparing the port against its own idea of what the input was.

`parity/pinned.py` refuses to export under a different `transformers` or
torch than the references were built with. Bump the pin and every
reference must be re-exported together — otherwise a version bump
silently invalidates them and looks like a phantom regression.

## Roadmap and study guide

- `ROADMAP.md` — the gate ledger, every step's verification, every
  mutation table, and an explicit **Out of scope** section.
- `STUDY.md` — the twelve decisions that were expensive here, written so
  the reasoning transfers to a different codebase, each with the failure
  that motivated it attached.
