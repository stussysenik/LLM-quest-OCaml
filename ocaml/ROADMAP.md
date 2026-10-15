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

## Deliberate simplifications

Stated as *what the port does not do*, with the reason, so each one can be
revisited rather than rediscovered.

- **Dropout is identity at inference.** Every shipped config sets `0.0`,
  and the reference exports run under `.eval()`.
- **No tokenizer.** Generation parity is gated on token IDs, which is the
  part the model owns; `Vlm.generate_cached` returns ids for the same
  reason.
- **No training.** No optimizers, LR schedules, AMP, gradient clipping,
  checkpoints written. `llm_quest`'s `*_training.py` scripts and
  `vit_engine.vit_training_eval_loop` / `vlm_engine.vlm_training_loop_simple`
  are out of scope. What *is* ported is the measurement side
  (`lib/loss.ml`, `Vlm.vlm_loss`, `Vit.batch_loss`), because a loss you
  cannot evaluate is a loss you cannot tell you got right.
- **Image resize is not torchvision's.** `lib/image.ml` implements
  `to_tensor` and ImageNet `normalize` exactly (both **bit-identical** to
  torch) and `resize_bilinear` at torch's `interpolate(align_corners=False)`
  sample positions (2.22e-16). `vlm_generation.py` calls
  `torchvision.transforms.functional.resize`, which antialiases and
  resamples in the integer domain. See `lib/image.ml`'s header.
- **`vlm_training.py` writes PyTorch `.pth` pickles.** This port reads
  safetensors only and has no pickle reader — a pickle is executable data.
  Converting the two files into the `vit.`/`adapter.`/`gpt.` namespace
  layout `Vlm_loader` reads is one `safetensors.torch.save_file` call away
  on the Python side.
- **Scope: `llm_quest/gpt`, `llm_quest/gpt_to_llama3`,
  `llm_quest/multimodal`.** The other twelve packages in `llm_quest/`
  (`moe`, `alignment`, `finetuning`, `reinforcement_pretraining`,
  `speculative_decoding`, `llama3_to_deepseekv3`, `llama3_to_gemma3`,
  `qwen`, `xiaomi`, `linear_attention_to_gated_deltanet`, `common`,
  `experimental`) are not ported. See "Out of scope" below.

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
| D1 | RMSNorm | ✅ Two references now, because `LlamaRMSNorm.forward` hard-casts to fp32. (i) vs that module, unpatched: 3.3e-7 / 7.8e-7 — which is **entirely the cast**, matched to every printed digit — with the bound *derived* from the measured fp32 island (2x it: 6.7e-7 / 1.6e-6) instead of the old written-down 1e-6 whose margin was 1.29x; (ii) vs `torch.nn.functional.rms_norm` in fp64: 4.4e-16 / 2.0e-14 at 1e-12. Mutation: rounding the port's output to fp32 **passes** (i) and **breaches** (ii) at 1.8e-7 / 3.4e-7 — the two gates do different jobs. Plus zero-mean identity vs LayerNorm exact (0.0); shift-sensitive where LayerNorm is shift-invariant; zero weight zeroes the output; eps placement changes the result (2.4e-6) both ways; `Config.norm_eps` family-specific (1e-6 Llama / 1e-5 GPT-2) and it is the value the op ran with. **15 gates** |
| D2 | SwiGLU | ✅ Two references. `LlamaMLP.forward` has **no** dtype cast — it computes in whatever it is handed — so its fp32 reference was an export choice, not the module's behaviour: it bounded the gate at 1e-5 while the port computes in fp64. So (i) vs the fp32 module, bound derived from the measured island (7.97e-7 / 1.77e-6 -> 1.6e-6 / 3.5e-6), and (ii) vs the same module in float64: 4.4e-16 / 2.2e-15 at 1e-12. Mutation: rounding the port to fp32 passes (i), breaches (ii). Also the same treatment for the silu grid (`d2.silu.ref64`). Mid is square on purpose so a transposed `down_proj` fails *silently* rather than raising; zero input zeroes the op and not the GPT-2 FFN (no bias); up path load-bearing; silu stable (naive sigmoid caught at -710); `Config.ffn_inner` is `intermediate_size` (128 ≠ 4×64 on tiny_test) and is the shape D2a's weights were checked against; GELU-on-the-gate changes the function. **19 gates**, all six mutations ran against the committed gates — each bites its own gate |
| D3 | RoPE, base | ✅ `lib/rope.ml`, `parity/export_rope.py`, `test/test_rope.ml`. D3a vs `RoPE.apply` (9.8e-8 / 8.6e-8 abs, fp32 ref — and fp32 is *the implementation*: `llm_quest/common/rope.py` asserts `dtype == torch.float32` and disables autocast for the angle computation, so unlike D2 this is faithful rather than an export choice); D3b relative-position invariance on the *same* q/k vectors at shifted positions (3.3e-16); D3c explicit 2D rotation matrix ≡ apply (exact); D3d `position_ids=[p]` ≡ full-forward row p (exact, 6.0e-8 vs ref). **9 gates**, 3 mutations run against them |
| D4 | RoPE, `llama3` scaling | ✅ `compute_angles_llama3` vs `_compute_llama3_parameters` on the frequency table alone (2.7e-7 / 5.0e-7, fp32 ref); scaling provably non-identity (7.9e-4) and provably differs from base RoPE (4.6e-4). **4 gates** added, 13 total |
| D5 | GQA | ✅ `parity/export_gqa.py`, `test/test_gqa.ml`, `GroupedQueryAttention` in `lib/attention.ml`. D5a vs `LlamaAttention` (eager), D5b vs this repo's own `llama_attention.py`; D5c the *anti*-gate — grouping is a permutation, so a repeat-tiling implementation must provably **disagree** (>1e-3) and the true one must be the closer match, which is the only gate that can catch an implementation that is accidentally right; D5d cache sized by `Config.kv_heads`, and a wide `num_heads` cache is **rejected** by both the cache constructor and `forward` rather than silently reshaped; D5e `num_repeat == 1` collapses GQA to plain MHA identically; D5f q/k/v carry no bias, `o_proj` does, and a zero bias ≡ no bias. **18 gates** |
| E | tiny-random Llama parity | ✅ **Complete.** `Llama.forward_hidden_states` composes D1+D2+D3/D4+D5; `parity.ml` gained `check_llama_layout` (Llama's own expectations, since K/V are `kv_heads * head_dim` wide, not `num_heads * head_dim`). `llama_ref` carries `hidden.0/1/2` + `logits`, and `make ci`'s `parity-llama-forward` gates **every one** plus the layout gate, at 1e-9. This row previously read "🔶 E1 only — the remaining stages land after E". That was true when written, stopped being true at `7290c0a`, and was not updated because nothing failed while it was wrong. |
| F | Llama-3.2-1B parity | ⬜ **Not started — out of scope for this pass.** 14 stages on real weights (~2.5 GB bf16), plus a `make ci-llama` target that does not exist in the Makefile. See "Out of scope" below. |
| G | Greedy generation parity | ✅ **Complete** — token-for-token vs `generate(do_sample=False)` as `parity_gen` (5 gates, cached + uncached paths both token-identical over 20 steps) |

## Multimodality (Parts 1 & 2 of the quest)

Same rule as above: each row gates the next, and every gate is checked
against a PyTorch export before it is composed with anything.

| Step | Piece | Gate |
| :--- | :--- | :--- |
| ViT | `lib/vit.ml` — patch embed, CLS token, learned position embedding, pre-norm encoder blocks, MHSA, MLP, layernorm, classification head | ✅ `parity/export_vit.py`, `test/test_vit.ml`. Every stage vs `vit_for_image_classification`; the CLS token is proven load-bearing (pooling the mean of patches diverges >1e-6) and `LayerNorm_make` defaults to eps-**outside**, the ViT convention. **25 gates** |
| VLM | `lib/vlm.ml` — ViT→LLM adapter (2-layer MLP) + early fusion | ✅ `parity/export_vlm.py`, `test/test_vlm.ml`. Fusion is *placement*: 17 vision embeddings then 6 text, never appended; the attention mask is 17 vision ones ++ the text mask; the shift for next-token logits slices at `n_vit - 1`, and both the off-by-one and the append-instead-of-place variants are caught by anti-gates. **30 gates**, run at two widths (the reference config and a deliberately simple 32→48 adapter) |

The VLM gates were mutation-tested as a batch — breaking early fusion four
ways (skip the adapter, append instead of place, off-by-one slice, drop
the vision mask) each takes `make ci` to exit 1. A gate set that has never
been observed to fail is an assumption.

### The second pass: from "gated" to "usable"

Parity above proved the *forward pass* was right. It did not make the
models **loadable**, **feedable** or **measurable** — and all three gaps
were invisible to every gate that existed, because each test built its own
weights by hand. Three rows close them.

| Step | Piece | Gate |
| :--- | :--- | :--- |
| M1 | `lib/vit_loader.ml`, `lib/gpt_repo_loader.ml`, `lib/vlm_loader.ml` + `parity/export_loader.py` — read a real fused checkpoint (`vit.` / `adapter.` / `gpt.` namespaces) | ✅ `test/test_loader_vlm.ml`, **26 gates**. Loading reproduces the reference activations for all three models at once (ViT 9.8e-15, adapter 2.6e-13, fused logits 1.8e-14); parameter counts match the file's element counts; `gpt.` needed its own loader because `vlm_training.py` builds the decoder from `llm_quest.gpt.GPTModel`, a **third** naming scheme alongside HF gpt2 and HF Llama |
| M2 | `lib/image.ml` — `to_tensor`, ImageNet `normalize`, `resize_bilinear` | ✅ `parity/export_image.py`, `test/test_image.ml`, **17 gates**. `to_tensor` and `normalize` are **bit-identical** to torch (0.000e+00); `resize_bilinear` at 2.22e-16 against torch's `interpolate(align_corners=False)` — *not* torchvision's antialiased resize, which the header and the test output both say. I-d is the anti-gate: on a 9×13 input an axis swap and a half-pixel-off resampler each still produce 16×16 output |
| M3 | `lib/loss.ml`, `Vlm.vlm_loss`, `Vit.batch_loss`/`batch_accuracy` | ✅ `parity/export_loss.py`, `test/test_loss.ml`, **27 gates**. The subject is `ignore_index`, not softmax: ignored rows leave the mean's denominator as well as its numerator, and an all-ignored reduction is NaN rather than 0.0 |
| M4 | Vision-prefix KV cache and the caption loop — `Gpt.prefill_from_emb_cached`, `decode_from_emb_cached`, `Vlm.prefill_cached`, `decode_step_cached`, `generate_cached` | ✅ G-a…G-f inside `test/test_loader_vlm.ml`, **6 gates**. Linear instead of the reference's per-step quadratic re-forward; G-b is the claim that they agree, checked per step against the reference's own per-step logits rather than through an argmax |

**Five things only a real checkpoint revealed.** All found by exporting a
real `state_dict` and pointing a loader at it, none by reading code:

| Finding | What a hand-written table gets wrong |
| :--- | :--- |
| `ViTAdapter`'s second `Linear` is at `Sequential` index **3**, not 2 — `Sequential(Linear, GELU, Dropout\|Identity, Linear)`, and no module in the gap carries a parameter | counting only the layers that *have* weights predicts index 2. The first draft claimed `2.weight` and failed on sight of the state dict |
| `ViTAdapter`'s linears are `bias=False` | the loader was reading biases that no checkpoint has |
| `GPTModel.out` is `bias=False` while every other Linear is not | ditto, in the other direction |
| `trf_blocks.N.att.mask` is a persistent **bool** buffer | the reader could not open a real file *at all* — F32/F64/I64 only |
| That mask stores `triu(ones, diagonal=1)`: **1 means masked** | the port's own `attn_mask` is a bool array where `true` means *attend*. Read straight through, it builds a bidirectional decoder that generates fluent nonsense and fails no shape check |

Plus a sixth found by a gate rather than a load: **`GPTModel.forward`
adds no positional embedding when `input_embedded=True`** — its
`x = x + pos_emb` sits *inside* `if not input_embedded:`. Adding positions
measures **1.36e-01** against the reference's logits (rel ~1e-2): enough
to wreck a caption, far too small for any structural check to notice.

**The loaders' mutation table, all three run against the committed gates:**

| Mutation | Caught by | Result |
| :--- | :--- | :--- |
| Square projection transposed (only where `d_in = d_out`, so `~expected_shape` cannot catch it) | L1 ×5 | 5 FAIL, exit 1 |
| Conv patch order reversed (kept in-shape) | L1 ×4 | 4 FAIL, exit 1 |
| `att.mask` polarity flipped | `Gpt_repo_loader.check_mask` | loader refuses, exit 2 |

**M4's mutation table:**

| Mutation | Caught by | Result |
| :--- | :--- | :--- |
| Positional embeddings re-added to prefill | G-a, G-b | 2 FAIL, exit 1 |
| Decode step embedding token-**with**-position instead of bare | G-b, G-c | 2 FAIL, exit 1 |

**D3/D4's mutation table, all three run against the committed gates:**

| Mutation | Caught by |
|---|---|
| `rotate_half` negation dropped | 7 gates (d3a/d3c/d3d, and the D4 tables inherit it) |
| Halves convention → interleaved (`j mod half` → `j / 2`) | 3 gates — the classic RoPE convention bug, caught numerically by D3a |
| llama3 smooth-interp terms swapped | `d4_cos` 3.8e-4, `d4_sin` 2.2e-2 against a 1e-5 gate |

D3b was **wrong when first written** — it compared `q(i)` against `q(i+d)` from different random rows, which tests nothing. The reference implementation failed it too (1.5e+01), which is what identified it: relative-position invariance is a statement about *one* q and *one* k at shifted positions, not about two unrelated vectors. Worth recording because the failure mode is a gate that can only ever fail, which is the same as no gate.

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

## Out of scope (stated, not implied)

So that "the roadmap is green" is not read as "the port is finished".

**Step F — Llama-3.2-1B parity on real weights.** ⬜ Not started. The
reasoning above still holds: tiny-random catches structural bugs and F
would test numerical behaviour under real weight distributions. Cost is
~2.5 GB bf16 plus ~7.5 GB of disk for the export, and it wants its own
`make ci-llama` target, which does not exist. Everything else in the
Llama 3.2 path — config, RoPE, GQA, RMSNorm, SwiGLU, loader, layout gate,
generation — is ported and gated.

**Twelve of the fifteen packages in `llm_quest/`.** Ported: `gpt`,
`gpt_to_llama3`, `multimodal`. Not ported: `moe`, `alignment`,
`finetuning`, `reinforcement_pretraining`, `speculative_decoding`,
`llama3_to_deepseekv3`, `llama3_to_gemma3`, `qwen`, `xiaomi`,
`linear_attention_to_gated_deltanet`, `common`, `experimental`. So the
quest's headline content — Mixture-of-Experts, RL/alignment, fine-tuning,
speculative decoding, the modern architectures — is unported. The line
GPT-2 → Llama 3.2 is finished; everything past it is untouched.

**Training.** Deliberate. See "Deliberate simplifications" above for what
was left behind along with it, and `lib/loss.ml` for the measurement side
that *was* brought across.

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
- **Reference activations are float64; weights stay float32** (that is
  what the checkpoints hold, and the loaders assert F32 against them).
  This supersedes the paragraph above, and the reason is that the
  criterion could not be calibrated by choosing a number.

  The old gate compared a faithful fp64 OCaml port against an **fp32**
  PyTorch reference at `rel_tol = 1e-6`, and breached at ~1.4e-6. The
  tempting reading is "the port has a bug"; it did not. Measured on this
  machine, PyTorch disagrees with **its own fp64 run** by 2.3e-6
  relative at block 3 and 2.0e-6 at the logits. So the gate was
  measuring the reference's rounding, and the number moves with the
  host's CPU BLAS reduction order — the same class of problem as the CUDA
  vs CPU torch note above, one level up.

  With the reference exported in float64, both sides sum in fp64 from
  bit-identical inputs and what remains is summation order:

  | stage | before (fp32 ref) | after (fp64 ref) |
  |---|---|---|
  | embeddings | 9.9e-6 | **0.0 exactly** |
  | block_0 | 3.5e-4 | 1.4e-13 |
  | blocks 1–10 | 3.6e-4 | 1.3e-12 … 2.3e-12 |
  | block_11_lnf | 3.6e-4 | 1.7e-13 |
  | logits | 2.4e-4 | 4.4e-13 |

  (The "after" column is what `make ci` prints on this machine; the error
  grows through the stack as expected and stays four orders under the
  1e-9 gate.)

  Every *parity* stage moved from 1e-6/1e-5 to **1e-9** — seven orders of
  magnitude tighter than the gate it replaces, and now the same number on
  any machine instead of a host-dependent one. The dual abs+rel criterion
  survives; what changed is that the reference stopped being the least
  accurate thing in the comparison.

  **Scope, because the distinction matters.** This covers the full-model
  stages — GPT-2, the Llama family, and the multimodal references — where
  the reference is an exported model run. The per-operator probes
  (D1…D5) are a different case and are now split by whether fp32 is the
  *implementation's* behaviour or the export's choice:

  | probe | fp32 reference is… | treatment |
  |---|---|---|
  | D1 RMSNorm | HF's own cast (`hidden_states.to(torch.float32)`) | kept unpatched, bound derived from the measured island; **plus** an fp64 gate vs torch's kernel |
  | D2 SwiGLU | only the export's choice — `LlamaMLP` has no cast | fp64 reference added, both gates ship |
  | D3 RoPE | the implementation's: `rope.py` asserts `float32` and disables autocast | kept; faithful |
  | D4 llama3 angles | the implementation's: `_compute_llama3_parameters` is fp32 | kept; faithful |
  | D5 GQA | the implementation's: HF eager attention + `finfo(float32).min` mask | kept; faithful |

  The D1 finding is worth stating plainly, because it is the same defect
  as the one above at a smaller scale: the gate read 7.767e-07 and the
  fp32 island is 7.767e-07 — *to every printed digit*. The gate was
  measuring HF's dtype policy, and its 1.29x margin against a
  written-down 1e-6 would have gone green→red on a host whose BLAS
  reduces in a different order. Deriving the bound from the island (and
  shipping the fp64 companion) is what makes that impossible.

- **A gate that cannot fail is not a gate.** D3b was written wrong the
  first time — it compared `q(i)` against `q(i+d)` from different random
  rows, which is not the statement RoPE makes. It failed on the reference
  implementation too (1.5e+01), which is what exposed it: the property is
  about *one* q and *one* k at shifted positions. Kept here because the
  failure mode — a gate that only ever fails — is indistinguishable from
  no gate at all, and it is easy to write.
