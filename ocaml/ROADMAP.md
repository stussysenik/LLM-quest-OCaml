# OCaml Port — Roadmap

The governing rule, from the review that set this order:

> **Verification before optimization.** Each step must be numerically
> verified against PyTorch before the next step begins. Never debug two
> things at once ("is my batching correct?" + "is my architecture
> correct?").

## Verified milestone (commit `ocaml/` initial)

- [x] GPT-2 single-sequence forward: attention → block → model → sampling
- [x] 26/26 self-consistency tests green (`test/test_all.ml`)
- [x] Causal invariant: row 0 of a 2-token forward == 1-token forward
- [x] Demo binary generates greedily (`bin/main.ml`)

These tests prove **self-consistency only** — not parity with PyTorch.

## Deliberate simplifications (unverified / missing)

- Single sequence only, no batch dim (`b x s x d`)
- Dropout is identity at inference (matches all shipped configs: `0.0`)
- `generate_simple_loop` only (no KV-cache / batched loops)
- No weight loading — random init only (`N(0, 0.02)`)

## Ordered plan (each step gates the next)

| # | Step | Verification gate |
|---|------|-------------------|
| 1 | **Weight loader (safetensors)** — minimal F32/I64 parser in stdlib OCaml, HF `gpt2` naming map (`transformer.h.N.*`, transposed Conv1D, `lm_head` transpose) | Embeddings match PyTorch to ~1e-6 |
| 2 | **Intermediate activation parity** — expose per-block hidden states (`output_hidden_states=True` gives 13 checkpoints) | Block outputs match to ~1e-5 |
| 3 | **Final logits parity** — single-sequence forward on loaded weights | Logits match to ~1e-4 (fp32 accumulation over 12 layers) |
| 4 | **Batched forward** — add batch dim to tensor ops | batch=1 == single-sequence; batch=N repeats == N identical outputs |
| 5 | **KV-cache generation loop** — `generate_loop_kv_cache` | Greedy decode **identical** with and without cache (catches off-by-one in mask/positions) |
| 6 | **Checkpoint loader + real text** — GPT-2 checkpoint format end to end | Token-by-token match with HuggingFace generation |

## Notes

- HF `gpt2` uses `gelu_new` (tanh approx), the repo uses exact-erf GELU.
  The loader must select the matching activation or parity will fail at
  every block past the first FFN.
- HF `Conv1D` weights are stored `(in, out)` — no transpose needed.
  `lm_head` (`nn.Linear`) is stored `(out, in)` — transpose needed.
- Reference export script lives in `ocaml/parity/` (Python side, needs
  torch+transformers, CPU is fine).
- transformers 5.x `GPT2Model` appends the **post-`ln_f`** state as
  `hidden_states[-1]` (verified: `hidden.12 @ lm_head.T == logits` to
  3e-5, and its std is ~5 vs ~80 for the raw outlier-dominated
  states). The parity harness compares it against our final-LN
  output, not the raw last-block output.
- GPT-2 activations have outlier dims (~3000), so block gates use a
  dual abs+rel criterion: fp32-vs-fp64 summation noise on those dims
  is ~4e-4 absolute (~1e-7 relative). A flat 1e-5 absolute gate is
  miscalibrated there — and the numbers prove it rather than the
  implementation being wrong.
