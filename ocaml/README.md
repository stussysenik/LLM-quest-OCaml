# LLM Quest — OCaml port

Pure-OCaml (stdlib only, no Owl/torch bindings) re-implementation of the
Python/PyTorch repo at the workspace root. Goal, per the project brief:
**SOTA of model understanding** — every op spelled out, zero magic.

## Layout

| OCaml | Python counterpart |
|---|---|
| `lib/config.ml` | `config.py` (GPT family) |
| `lib/tensor.ml` | `torch` ops used across `llm_quest/` |
| `lib/norm.ml` | `LayerNorm`, `GELU` in `llm_quest/gpt/gpt_transformer_block.py` |
| `lib/kv_cache.ml` | `KVCache` in `llm_quest/utils.py` |
| `lib/attention.ml` | `llm_quest/gpt/gpt_attention.py` (v1/v2/v3 + `MultiHeadAttention`) |
| `lib/block.ml` | `FFN`, `TransformerBlock` in `gpt_transformer_block.py` |
| `lib/gpt.ml` | `llm_quest/gpt/gpt_model.py` (`GPTModel`) |
| `lib/sampling.ml` | `sampling*` + `generate_simple_loop` in `llm_quest/generate.py` |
| `bin/main.ml` | PART A demo in `generate.py __main__` |
| `lib/sampling.ml` | `sampling*` + `generate_simple_loop` in `llm_quest/generate.py` |
| `lib/safetensors.ml` | HF safetensors format (shared by both loaders) |
| `lib/gpt2_loader.ml` | HF `gpt2` naming map (`Conv1D`, `ln_f`, `lm_head`) |
| `lib/llama.ml` | Llama weight record — shapes only, forward lands in step D |
| `lib/llama_loader.ml` | HF `LlamaForCausalLM` naming map (`nn.Linear`, RMSNorm, SwiGLU) |
| `test/test_all.ml` | `__main__` checks in `gpt_attention.py` / `generate.py` |
| `test/parity*.ml` | PyTorch parity harnesses (activations, KV, generation) |
| `test/test_loader.ml` | Loader gates C1-C4 + the `[out, in]` layout gate |

## Build

With dune (once installed):

```bash
cd ocaml && dune build && dune exec ./bin/main.exe && dune test
```

Without dune (stock OCaml 4.14, what CI uses today):

```bash
cd ocaml
ocamlopt -I lib -o test_all \
  lib/tensor.ml lib/config.ml lib/norm.ml lib/kv_cache.ml \
  lib/attention.ml lib/block.ml lib/gpt.ml lib/sampling.ml \
  test/test_all.ml
./test_all
ocamlopt -I lib -o demo \
  lib/tensor.ml lib/config.ml lib/norm.ml lib/kv_cache.ml \
  lib/attention.ml lib/block.ml lib/gpt.ml lib/sampling.ml \
  bin/main.ml
./demo
```

## Deliberate simplifications (documented in code)

- Dropout is identity at inference (all shipped configs use `0.0`).
- No tokenizer. Generation parity is gated on **token IDs**, not decoded
  text — the reference file carries the tokenized prompt, so both sides
  start from identical input.

`config.ml` carries both families in one record: shared fields at the
top level, everything that differs behind `arch` (`Gpt2` / `Llama`).
Code that needs a shape reads an accessor — `Config.max_positions`,
`norm_eps`, `ffn_inner`, `kv_heads`, `head_dim`, `has_learned_positions`
— so shared code (block stack, attention, cache, loader) never matches
on family. Llama-only knobs (`rope_theta`, `rope_scaling`,
`tie_word_embeddings`) raise on a GPT-2 config rather than returning a
plausible-looking default.

## Running everything

`make ci` is the one-command green-tree proof: 60 unit gates + 15
loader gates + 14 PyTorch parity stages + a 12-layer KV cross-check +
5 generation gates + the step-D boundary check, about two minutes on
one CPU core.

```bash
cd ocaml && make ci
```

The PyTorch stages need reference files generated once against the
pinned stack in `parity/requirements.txt` (CPU is fine):

```bash
python -m venv /tmp/parity-env && /tmp/parity-env/bin/pip install -r parity/requirements.txt
/tmp/parity-env/bin/python parity/export_reference.py --family gpt2  /data/tmp/ocaml-parity/gpt2_ref.safetensors
/tmp/parity-env/bin/python parity/export_reference.py --family llama /data/tmp/ocaml-parity/llama_ref.safetensors
/tmp/parity-env/bin/python parity/export_kv.py       /data/tmp/ocaml-parity/gpt2_kv_ref.safetensors
/tmp/parity-env/bin/python parity/export_gen.py      /data/tmp/ocaml-parity/gpt2_gen_ref.safetensors
```

`export_reference.py` is one script for both families: `--family` picks
the model class, tokenizer and pinned revision from a table, and the
shared body probes the `hidden_states[-1]` convention (post-final-norm
state, by pairing it against the logits) and prints it, along with
`tie_word_embeddings`, before writing. A convention difference surfaces
at export, not as a phantom numerical breach at parity time.

`parity/pinned.py` refuses to export against a different `transformers`
than the committed references were built with — otherwise a version bump
would silently invalidate them and look like a phantom regression. The
pin covers both families: bump it, and re-export every reference.

## Roadmap

Full gate sequence in `ROADMAP.md`. The parity harness is parameterized
by family (`./parity_check --family {gpt2,llama}`; a family supplies a
loader, and `run`'s stage loop has no family branch). Next: split the
loader, then the Llama deltas — RMSNorm, SwiGLU, RoPE (base +
`llama3` scaling), GQA — each gated in isolation before composition,
then parity on `tiny-random-LlamaForCausalLM`. Real Llama-3.2-1B
weights are an optional final check, not a prerequisite.
