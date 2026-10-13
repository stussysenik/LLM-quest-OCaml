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

At that commit these tests proved **self-consistency only**. Steps 1-6
below replaced that with PyTorch parity.

## Deliberate simplifications (unverified / missing)

- Dropout is identity at inference (matches all shipped configs: `0.0`)
- No tokenizer — generation parity is gated on token IDs, which is the
  part the model owns
- `gpt2` (HF `gpt2`) only; the Llama family is the next port

## Ordered plan (each step gates the next)

| # | Step | Verification gate |
|---|------|-------------------|
| 1 | **Weight loader (safetensors)** — minimal F32/I64 parser in stdlib OCaml, HF `gpt2` naming map (`transformer.h.N.*`, transposed Conv1D, `lm_head` transpose) | ✅ Embeddings match PyTorch to ~1e-6 |
| 2 | **Intermediate activation parity** — expose per-block hidden states (`output_hidden_states=True` gives 13 checkpoints) | ✅ Block outputs match to ~1e-7 rel (dual abs+rel gates) |
| 3 | **Final logits parity** — single-sequence forward on loaded weights | ✅ Logits match to ~5e-5 |
| 4 | **Batched forward** — batch dim over shared single-sequence kernels | ✅ batch=1 ≡ single (bit-exact); repeats identical; padding+positions; per-sequence last-token; greedy decode parity |
| 5 | **KV-cache generation loop** — prefill + decode steps over `Kv_cache` | ✅ K1 cache ≡ no-cache tokens; K2 cache ≡ full-forward K/V; K3 batched ≡ single; K4 EOS freeze; K5 positions; plus cross-check vs PyTorch `past_key_values` (12/12 layers OK) |
| 6 | **Checkpoint loader + real text** — GPT-2 checkpoint format end to end | ✅ Greedy generation token-for-token vs `generate(do_sample=False)`: 20/20 exact (cached), 5-token prefix exact (simple loop), plus HF's stop contract |

## Llama 3.2 port

Same governing rule: verify each delta against PyTorch before composing
it with the next. `make ci` stays green after every row.

| Step | Delta | Gate |
| :--- | :--- | :--- |
| A | `config.ml` → one `t` for both families, family detail behind `arch`, accessors (`max_positions`, `norm_eps`, `ffn_inner`, `kv_heads`, `head_dim`, `has_learned_positions`) | ✅ GPT-2 parity **bit-identical** (every `max_abs_diff` unchanged to the digit); 15 config gates |
| B | `test/parity.ml` → parameterized; export scripts family-tagged | export + parity green for GPT-2 via the new path |
| C | `hf_loader.ml` split → `gpt2_loader.ml` + `llama_loader.ml` (`safetensors.ml` stays shared) | GPT-2 parity green via the new loader |
| D1 | RMSNorm | vs `LlamaRMSNorm` on random input (1e-6, tiny + `hidden=2048`); eps-inside-sqrt; differential vs GPT-2 LayerNorm (no mean subtraction, no bias) |
| D2 | SwiGLU | vs `LlamaMLP`: `down(silu(gate(x)) * up(x))`, width = `intermediate_size` |
| D3 | RoPE, base | relative-position invariance `q·k(i,j) == q·k(i+d,j+d)`; direct vs `apply_rotary_pos_emb`; cached-at-pos-p ≡ full-forward row p |
| D4 | RoPE, `llama3` scaling | vs `_compute_llama3_parameters` on the frequency table alone, before any attention |
| D5 | GQA | vs `LlamaAttention`; KV cache sized by `Config.kv_heads`, not `num_heads` |
| E | tiny-random Llama parity | 14 stages against `hf-internal-testing/tiny-random-LlamaForCausalLM` |
| F | Llama-3.2-1B parity | 14 stages on real weights — **optional**, see below |
| G | Greedy generation parity | token-for-token vs `generate(do_sample=False)` |

Steps A-E need no real weights: random tensors and the ~5MB
tiny-random checkpoint. The tokenizer (Llama 3 tiktoken BPE) is a
separate project — all parity gates use token IDs directly, so it is not
on the critical path for model correctness.

**Step F is a final exam, not a prerequisite.** tiny-random catches every
structural bug (layout, transpose, head grouping, RoPE convention, eps
placement). What it cannot catch is numerical behaviour under real
weight distributions — GPT-2's ~3000-magnitude outlier dims are exactly
why the dual abs+rel gate exists, and random weights have none. Llama 3
has its own "massive activations", so the same class of bug is plausible.
Run it when convenient; a port without it is still verified, just not
verified under real numerical conditions. Cost: ~2.5GB bf16 weights,
~7.5GB disk with the export, minutes per forward on CPU — hence a
separate `make ci-llama` target rather than the main `make ci`.

## Notes

- HF `gpt2` uses `gelu_new` (tanh approx), the repo uses exact-erf GELU.
  The loader must select the matching activation or parity will fail at
  every block past the first FFN.
- HF `Conv1D` weights are stored `(in, out)` — no transpose needed.
  `lm_head` (`nn.Linear`) is stored `(out, in)` — transpose needed.
- Reference export script lives in `ocaml/parity/` (Python side, needs
  torch+transformers, CPU is fine). `parity/requirements.txt` pins the
  stack (`transformers==5.19.0`, `torch==2.14.1+cpu`,
  `safetensors==0.8.0`, `gpt2@607a30d7`) and `parity/pinned.py` refuses
  to run against anything else. Without the pin a `pip install -U
  transformers` invalidates the committed references and the resulting
  gate failure reads as a phantom OCaml regression.
- Generation parity (step 6) compares **token IDs, not decoded text** —
  the OCaml side has no tokenizer by design, and the reference file
  carries the tokenized prompt so both sides start from identical input.
  A tokenizer is its own project (see Llama notes), not a prerequisite
  for model correctness.
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
