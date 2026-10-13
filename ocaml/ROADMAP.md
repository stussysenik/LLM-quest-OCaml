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
| B | `test/parity.ml` → parameterized (`run ~family`, family-erased `runnable`); `export_reference.py --family {gpt2,llama}` | ✅ B1 GPT-2 parity **bit-identical** through the new path; B2 `make ci` gate count unchanged, 101.9s → measured; B3 `--family llama` export runs, prints the `hidden_states[-1]` convention + `tie_word_embeddings`; B4 `run`'s body has no family branch |
| C | `hf_loader.ml` split → `gpt2_loader.ml` + `llama_loader.ml` + `llama.ml` (`safetensors.ml` stays shared, gains `get_tensor`/`require_absent`/`assert_exact_keys`) | ✅ C1 key set + every shape match the config; C2 loaded params == file's 1032272; C3 three wrong-config variants each raise naming the tensor; C4 (15 gates total) tie aliases (physically, `==`) / untied-without-head raises / tie-with-head raises; C5 `--family llama` loads weights then stops at step D; GPT-2 parity **bit-identical** through the rewritten loader |
| L | **Weight layout → `[out, in]` project-wide.** `Attention.linear.w` becomes `(out, in)`; `Tensor.linear_fwd` is the one linear op and has no layout parameter; `Gpt2_loader` transposes `Conv1D` from `(in, out)` at load; `Llama_loader` unchanged (already `nn.Linear` order) | ✅ GPT-2 parity **bit-identical** — every `max_abs_diff` unchanged to the digit, which is the proof the refactor computes the same function; from-scratch weights *and* logits bit-identical as well; new gate `layout_gpt2_out_in` / `layout_llama_out_in` re-derives each `out` from the config, covering the square projections shape assertions cannot |
| D1 | RMSNorm | ✅ vs `LlamaRMSNorm` at hidden 16 **and** 2048 (3.3e-7 / 7.8e-7, dual abs+rel 1e-6); zero-mean identity vs LayerNorm exact (0.0); shift-sensitive where LayerNorm is shift-invariant; zero weight zeroes the output; eps placement changes the result (2.4e-6) both ways; `Config.norm_eps` family-specific (1e-6 Llama / 1e-5 GPT-2) and it is the value the op ran with. **13 gates**, each proved to bite by mutating the op |
| D2 | SwiGLU | ✅ vs `LlamaMLP` at hidden 16/inter 64 and the square 256/256 (8.0e-7 / 1.8e-6, dual abs+rel 1e-5 — the reference is fp32; mid is square on purpose so a transposed `down_proj` fails *silently* rather than raising); zero input zeroes the op and not the GPT-2 FFN (no bias); up path load-bearing; silu stable (naive sigmoid caught at -710); `Config.ffn_inner` is `intermediate_size` (128 ≠ 4×64 on tiny_test) and is the shape D2a's weights were checked against; GELU-on-the-gate changes the function. **19 gates**, all six mutations ran against the committed gates — each bites its own gate |
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

### Reference files

| File | Family | Exported by |
| :--- | :--- | :--- |
| `/data/tmp/ocaml-parity/gpt2_ref.safetensors` | gpt2 (652MB, real weights) | `parity/export_reference.py --family gpt2` |
| `/data/tmp/ocaml-parity/gpt2_kv_ref.safetensors` | gpt2 | `parity/export_kv.py` |
| `/data/tmp/ocaml-parity/gpt2_gen_ref.safetensors` | gpt2 | `parity/export_gen.py` |
| `/data/tmp/ocaml-parity/llama_ref.safetensors` | llama (`hf-internal-testing/tiny-random-LlamaForCausalLM`, ~5MB) | `parity/export_reference.py --family llama` |
| `/data/tmp/ocaml-parity/llama_tied_ref.safetensors` | llama, head tied (same weights, no `lm_head.weight`) | `parity/export_reference.py --family llama --drop-head` |
| `/data/tmp/ocaml-parity/norm_ref.safetensors` | RMSNorm inputs + `LlamaRMSNorm` outputs (no model weights) | `parity/export_norm.py` |
| `/data/tmp/ocaml-parity/mlp_ref.safetensors` | SwiGLU inputs/weights + `LlamaMLP` outputs + the silu extreme grid (no model weights) | `parity/export_mlp.py` |

- `parity/requirements.txt` installs torch **separately**, from the
  CPU index only. `--index-url` inside a requirements file is a global
  option, so the old one-line form could not work; and leaving it off
  let PyPI serve the CUDA build. `pinned.py` now enforces
  `torch.__version__ == "2.14.1+cpu"` as well as the transformers pin —
  a CUDA torch reduces in a different order, so a reference exported on
  one and gated against the other looks like a numerical disagreement
  that is really an environment mismatch.
- **Weight layout is `[out, in]`, project-wide.** Every matrix a loader
  hands the kernels has its output features as *rows*, matching
  `nn.Linear` and `Tensor.linear_fwd w x = x @ w^T`. It is that way
  because it is the layout most of the world already stores: the Llama
  loader is a pure read, and the GPT-2 loader converts HF's `Conv1D`
  from `(in, out)` at load. The alternative — one convention per family —
  means every call site in the attention and MLP code picks, and a wrong
  pick is silent: for both `gpt2` and tiny-random-Llama,
  `hidden = heads * head_dim`, so every attention projection is square
  and a transposed weight multiplies to plausible numbers.
  `mat_mul`'s inner-dimension check catches the non-square cases; the
  square ones are the job of the layout gate (`layout_gpt2_out_in`,
  `layout_llama_out_in`), which re-derives each matrix's expected `out`
  from the config *independently* of the loader's own `~expected_shape`.
- **A normalization convention is a name, not a parameter.** `norm.ml` has
  one kernel, `norm_forward`, taking `~centering` (`Centering` /
  `No_centering`) and `~eps_placement` (`Eps_inside_sqrt` /
  `Eps_outside_sqrt`) as named variants; `layer_norm` and `rms_norm` state
  theirs and are the only two things the rest of the repo calls.
  `rms_norm` takes no convention flag at all — no centering, no shift, eps
  inside the sqrt *is* what the name means. A boolean that inverts at every
  call site is the thing someone flips without reading, and the eps
  difference between the two conventions is ~5e-6 relative: invisible in
  training, fatal for a 1e-6 parity gate. Renaming `eps_inside_sqrt` to the
  variant was mechanical and **bit-identical** — all 14 GPT-2 parity
  `max_abs_diff` values are unchanged to the digit, which is the proof the
  refactor computes the same function and not merely a similar one.
- **D1's differential gate is the one that matters.** An RMSNorm that
  quietly kept the mean subtraction breaches D1a on most inputs — but
  *passes* it on any input whose mean is ~0, and passes D1b's zero-mean
  identity by construction (that identity is what says the mean
  subtraction is the only difference). What catches it is shift
  *sensitivity*: LayerNorm is shift-invariant by definition and RMSNorm is
  not, so a shift-sensitivity gate cannot be satisfied by a LayerNorm.
  All four mutations were run against the committed gates to confirm each
  one bites: keeping the mean subtraction, moving eps outside the sqrt,
  adding a bias, and crossing `Config.norm_eps` (which D1a catches
  numerically at 4e-5, so D1d is not merely self-referential).
- `embed_tokens` and a tied `lm_head` are `[vocab, hidden]` — the
  embedding table's own layout, which coincides with `[out, in]` for the
  head. Tying therefore needs no transpose, and C4a pins that the tied
  head *is* the embedding matrix rather than a copy or its transpose.
- `tiny-random-LlamaForCausalLM` has `num_key_value_heads ==
  num_attention_heads` (4/4) — **no GQA**. Step E therefore does not
  cover GQA; `Config.llama_tiny_test_config` (4 heads / 2 kv heads)
  and step D5 do. Do not read a green E as a green GQA.

- **D2's mutation table, all six run against the committed gates:**

  | Mutation | Caught by |
  |---|---|
  | GELU instead of silu on the gate | D2a tiny+mid (3.4e-1 / 4.0e-1), D2e (diff 0.0) |
  | Naive sigmoid `1/(1+exp(-x))` | D2c's -710 gate **only** — in fp64 the naive form *is* the stable one to ~1e-16 everywhere except below the exp-overflow boundary (~-709.78), where it collapses to -0.0 against the true ~-7e-306. The gate pins the fp64 kernel to the *function*, stricter than the fp32 reference (which also flushes it to -0.0) — the same reasoning as `rms_norm` accumulating in fp64. |
  | Bias on `down_proj` | D2b's zero-input gate (+ D2a). A bias on `gate_proj`/`up_proj` is *masked* at zero input — the other path multiplies it by zero — and is caught by D2a numerically instead. The asymmetry is the gate's finding, not a hole: only a down-proj bias survives a zero input. |
  | `gate_proj` ↔ `up_proj` swapped | D2a tiny+mid — silu is not commutative under swap, and the export's independent random weights make the two orders differ (2.4e-1 / 1.8e-1). Verified that the chosen inputs distinguish them, as the plan required. |
  | `down_proj` transposed (reads `hidden` as `d_in`) | D2a mid *silently* (3.7e-1 — a square down_proj multiplies to plausible numbers), tiny *loudly* (inner-dim raise). Both observed; mid is square on purpose. |
  | `ffn_inner` = `4 * hidden` for Llama | D2d (`tiny_test`: 128 ≠ 256). The `tiny_random` config cannot catch this — 4×16 = 64 coincides — which is why D2d asserts the distinguishable config. |

- HF `gpt2` uses `gelu_new` (tanh approx), the repo uses exact-erf GELU.
  The loader must select the matching activation or parity will fail at
  every block past the first FFN.
- HF `Conv1D` weights are stored `(in, out)` and are **transposed** at
  load into the project convention `(out, in)`. `lm_head` (`nn.Linear`)
  is stored `(out, in)` already and is read as-is.
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
