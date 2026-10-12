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
| `test/test_all.ml` | `__main__` checks in `gpt_attention.py` / `generate.py` |

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

- Single batch / single sequence (no `(b, s, d)` batching yet).
- Dropout is identity at inference (all shipped configs use `0.0`).
- `generate_simple_loop` only; batched + KV-cache generate loops come
  with the batched forward.
- Weight init is GPT-2 style `N(0, 0.02)`; no pretrained-weight loader yet.

## Roadmap

1. Batched forward (`b x s x d`) + `generate_loop` / `generate_loop_kv_cache`.
2. Weight loader for the `gpt_s` checkpoint (`gpt_download_weights.py`).
3. `gpt_to_llama3/` (GQA, RoPE, RMSNorm, SwiGLU), then MoE / GRPO.
