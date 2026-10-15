# GOALS — session log for the OCaml port

## DONE — multimodality complete, env restored, `make ci` green

The goal below was met and then extended. Current state, measured:

- `cd ocaml && make ci` exits 0 — **273 PASS across 14 suites**, from a
  clean tree, on the pinned venv at `/data/tmp/venv`
  (torch 2.14.1+cpu, transformers 5.19.0, safetensors 0.8.0;
  `pinned.check_pinned()` passes).
- `PARITY_DIR` is back to the Makefile default `/data/tmp/ocaml-parity`
  and works as-is. **The `PARITY_DIR=/tmp/...` override and the "SIP
  makes /data read-only" note are obsolete** — that was a macOS machine.
  This one is Linux, `/data` is writable (141 GB free), and `/tmp` does
  not survive between sessions. All 12 references were re-exported
  fp64-current under `/data/tmp/ocaml-parity`.
- Toolchain is **OCaml 4.14.1, not 5.5.1**, and **dune is not
  installed**. The Makefile is the real build (`ocamlopt` directly) and
  the whole suite compiles warning-free under 4.14.1 — so `dune build`,
  last fixed on 2026-10-14, is *unverified here* rather than known-broken.
  Nothing in `make ci` depends on dune.

What this pass added on top of the original multimodality goal:

- `lib/vit_loader.ml`, `lib/gpt_repo_loader.ml`, `lib/vlm_loader.ml` —
  the models could be *gated* but not *loaded*; now they read a real
  fused checkpoint. 26 gates.
- `lib/image.ml` — `to_tensor` and ImageNet `normalize` bit-identical to
  torch, `resize_bilinear` at 2.22e-16 (not torchvision's antialias).
  17 gates.
- Vision-prefix KV cache + `Vlm.generate_cached` — linear instead of the
  reference's per-step quadratic re-forward. 6 gates.
- `lib/loss.ml`, `Vlm.vlm_loss`, `Vit.batch_loss`/`batch_accuracy`.
  27 gates.
- `Norm.erf` replacing Abramowitz & Stegun 7.1.26: 2.11e-07 → 1.11e-15
  on erf. Six gates moved 1e-6 → 1e-9.
- `STUDY.md` — twelve transferable patterns with the failure that
  motivated each.
- ROADMAP rows E/F/G corrected; the multimodality section gained M1–M4;
  an explicit "Out of scope" section now records step F and the twelve
  unported `llm_quest` packages, so a green roadmap is not read as a
  finished port.
- A **proofreading pass** (2026-10-15): dead code removed, four vacuous
  gates made real (one deleted), a temp-file leak fixed, every stale
  header/doc claim trued up. Net 274 → 273 gates, the delta being the
  deleted ref-vs-ref rope check. `make clean && make ci` exit 0.

## LOG

- 2026-10-13: Goal created. Beginning autonomous ralph loop on OCaml port.
- 2026-10-13: OCaml 5.5.1 + dune 3.24.2 installed. Baseline verified: test_all 42/42, test_batched 9/9, test_kv 9/9.
- 2026-10-13: Pinned parity venv setup (torch 2.14.1+cpu, transformers 5.19.0). References exported: norm, mlp, rope, gpt2, llama, llama_tied.
- 2026-10-13: D3 (RoPE base) complete — 9/9 gates green (lib/rope.ml, parity/export_rope.py, test/test_rope.ml).
- 2026-10-13: D4 (RoPE llama3 scaling) complete — 13/13 gates green incl. D4 (parity/export_rope_llama3.py). Three mutations run against the committed gates: rotate_half sign flip (7 gates), interleaved-vs-halves convention (3 gates), llama3 smooth-interp swap (3 gates, caught numerically at 3.8e-4 / 2.2e-2).
- 2026-10-13: Environment note — /data is read-only (macOS SIP); references live in /tmp/ocaml-parity. Makefile REF_* are `?=` so they override cleanly. pinned.py now accepts torch `2.14.1` as well as `2.14.1+cpu` (current CPU index dropped the +cpu suffix; torch.version.cuda is None so the build is CPU as pinned).
- 2026-10-13: **SIGBUS root cause found.** `export_gen.py` died with exit 138 (SIGBUS, not a Python exception) at every single-token forward. Cause: HF loads safetensors via mmap, and gpt2's `lm_head.weight` (tied to `wte`) comes back with `data_ptr() % 64 == 19`. torch's CPU BLAS picks a vectorised GEMV path for single-token forwards that needs 64-byte-aligned weights and dereferences it anyway. seq_len>=2 (GEMM) was fine, which is why every earlier export succeeded. Fix: `pinned.align_parameters()` clones params into aligned storage — bit-identical values, so references stay valid. Applied to all four checkpoint-loading exports.
- 2026-10-13: **GPT-2 parity was miscalibrated, not broken.** Against an fp32 reference the gates breached at rel~1.4e-6 (tolerance 1e-6). Decisive check: PyTorch disagrees with *its own fp64 run* by rel 2.3e-6 at block 3 — i.e. the reference's own fp32 rounding exceeded the gate, and it shifts with the host BLAS. Fix: export reference activations in float64 (weights stay F32 as checkpointed), add `Safetensors.mat_of_f64`, tighten every gate from 1e-6/1e-5 to **1e-9**. Result: embeddings exact, blocks ~1e-13, logits 1.7e-13 — a gate 7 orders tighter and now platform-independent.
- 2026-10-13: D5 (GQA) complete — 18/18 gates green (lib/attention.ml `gqa`, parity/export_gqa.py, test/test_gqa.ml). HF `LlamaAttention` and the repo's own `GroupedQueryAttention` agree to 0.000e+00 on identical weights, so the target is unambiguous. Two mutations: contiguous→tiled grouping (4 gates; `d5c_tiling_must_not_agree` goes to exactly 0.000, the signature of that specific mistake) and RoPE applied to V (2 gates).
- 2026-10-13: **`make ci` exits 0** — 132 PASS across 9 suites: 42 unit + 9 batched + 9 KV + 16 loader + 13 norm + 19 mlp + 13 rope(D3/D4) + 18 GQA(D5) + 5 generation, plus 14 parity stages, 12-layer KV cross-check, and the C5 llama boundary check.
- 2026-10-13: **Step E complete** — `Llama.forward_hidden_states` composes D1+D2+D3/D4+D5; `parity.ml` gained `check_llama_layout` (Llama's own expectations, since K/V are `kv_heads * head_dim` wide, not `num_heads * head_dim`). All 14 stages green at the 1e-9 gate: embeddings exact, block_0 6.1e-13, block_1_lnf 1.2e-10, logits 1.7e-11.
- 2026-10-13: Second fp32 island found, in HF's own code: `LlamaRMSNorm.forward` hard-casts to `float32` regardless of model dtype, worth 1.26e-09 / 1.87e-07 / 2.20e-08 at block_0 / block_1 / logits — the exact three numbers the gate reported, which is how it was identified. `force_fp64_reference()` neutralises it at export. (Ruled out first: HF's eager attention pins softmax to fp32 too, but that contributes exactly 0.0 here.)
- 2026-10-13: C5's step-D boundary gate is retired — it asserted `--family llama` *fails*, which is no longer true. Replaced by `parity-llama-forward`, which asserts it exits 0 *and* that the layout gate ran *and* that a logits stage was reached, so a harness that quietly did nothing cannot pass.
- 2026-10-13: Makefile now takes a single `PARITY_DIR` instead of ten REF_* overrides (`make ci PARITY_DIR=/tmp/ocaml-parity`); the per-file vars stay `?=` overridable.
- 2026-10-13: **Multimodal Part 1 (ViT) complete** — `lib/vit.ml` (PatchEmbedding2D conv + CLS + pos-emb + encoder attention + CLS classification head), `parity/export_vit.py`, `test/test_vit.ml`. **25/25 gates green.** V-a patch embeddings are *exact* (0.000e+00 at 8x8/patch2; 4.4e-16 at 6x6/patch3).
- 2026-10-13: ViT specifics worth keeping — (a) `nn.Conv2d` is the ONE weight in the repo outside the `[out, in]` linear convention, kept as `[emb, C*P*P]` kernels so the layout is visible; (b) ViT's LayerNorm is `(x-mean)/(std+eps)` — eps **outside** the sqrt, the opposite of HF's, and at 1e-9 that is not a detail; (c) patch order is row-major over the (H/P, W/P) grid, and the positional table is indexed against it, so a mix-up yields a self-consistent model that matches nothing.
- 2026-10-13: Two ViT mutations run: column-major patch order (10 gates) and eps-inside-the-sqrt (V-f breaches at 1.2e-5 / 7.0e-5, two orders over the gate). The eps mutation exposed a **gate that could not fail** — V-e's "norms use Eps_outside_sqrt" compared the *test's own* record against a hardcoded constant and stayed green under mutation. Replaced with one that reads the library default.
- 2026-10-13: GELU accuracy measured, not assumed. `Norm.erf_approx` (Abramowitz & Stegun 7.1.26) is **2.11e-07** on GELU over [-12,12] — the old comment's "~1.5e-7" was erf's own bound and understated GELU's by ~40%. Gate V-g pins it; V-f runs at 1e-6 because of it, while V-a (no GELU) stays at 1e-9. A proper double-precision erf is the outstanding accuracy item.
- 2026-10-13: Found in the reference: `ViTModel.forward` asserts `x.shape[2] == img_width` on an NCHW tensor, so dim 2 is height — the Python only runs on square images. Recorded, not "fixed": the port mirrors the reference.
- 2026-10-13: `make ci` exits 0 — 149 PASS (was 132; +17 ViT), no warnings, all parity stages green.
- 2026-10-14: **Multimodal Part 2 (VLM) complete** — `lib/vlm.ml` (`adapter` = `Simple`/`Ffn`, `fuse`, `fuse_masks`, `fused_embeddings`, `forward_hidden_states`, `shifted_text_logits`), `parity/export_vlm.py`, `test/test_vlm.ml`. **30/30 gates green, `make ci` exit 0 from a clean tree** (199 PASS across 11 suites: 42+9+9+16+13+19+13+18+25+30+5).
- 2026-10-14: The reference is `ViTModel` + `ViTAdapter` + `GPTModel` from `llm_quest.multimodal`, wired exactly as `vlm_engine.py` wires them: `cat([vision, text], dim=1)` — vision FIRST — with `cat([ones(n_vision), text_mask])`. Two cases exported ("ffn" and "simple", the two branches of `ViTAdapter.__init__`). Export is byte-reproducible across runs (identical sha256).
- 2026-10-14: **The ViT and the LLM deliberately have different widths (32 -> 48).** A same-width pair would let an identity adapter pass — the adapter *is* a projection, so the export has to make "skipped" unrepresentable. It is also why a literal "skip the adapter" mutation cannot typecheck; the mutations that stand in for it are the ones below.
- 2026-10-14: Four VLM mutations, each caught with exit 1, each reverted: (1) fusion order reversed -> 6 failures (V2-b + V2-e, both cases); (2) `adapter_forward` hardcodes tanh GELU, ignoring the stored activation -> 3 failures, **ffn case only**, and V2-j stays green — which is the point: V2-j reads the stored variant, so only the behavioural gates catch a library that discards it; (3) `shifted_text_logits` starts at `n_vit` -> 4 failures (V2-f, both cases); (4) `fuse_masks` ignores the text mask -> 6 failures (V2-d + V2-e).
- 2026-10-14: Two gate-design bugs found and fixed by running them. **Tolerance:** the simple adapter was gated at 1e-9 because it has no GELU — but the ViT feeding it does, and that ~2e-8 is upstream of every case; the bound was re-gating V-g of test_vit and failing on it, so V2-a/b/e all run at 1e-6 while V2-c (token+position, never touches the ViT) stays at 1e-9 and measures 0.000e+00. **Slice count:** the test expected `text_len - 1`; `vlm_loss` slices `logits[:, nv-1:-1]` = `seq - nv` rows = `text_len`, not `text_len - 1` — the last *logit* row is dropped, not the last *text* row.
- 2026-10-14: Test robustness — `check_diverges`/`gate` now branch on a shape mismatch instead of raising `Failure("row count mismatch")`. Mutation 3 previously killed the suite mid-run and hid the remaining gates; it now reports 4 FAILs and exits 1 with the rest of the suite intact.
- 2026-10-14: Three real bugs caught before the reference was ever exported, all in `export_vlm.py`: (a) `hash(adapter_type)` for the seed — Python salts string hashes per process (`PYTHONHASHSEED`), which would have made the reference differ between runs; replaced by a case-index offset, then verified byte-identical across two runs; (b) `list(adapter.adapter)` on an ffn adapter yields `[Linear, GELU, Linear]`, so reading `.weight` blew up on the activation — factored into `adapter_linears`; (c) **ViT and GPT weights both wrote `{p}.block{i}` and `{p}.final_{scale,shift}`** — the second silently clobbered the first, which would have produced a reference that compiled and passed its own gates with GPT block 0's norms being the ViT's. Both stores now prefix `vit.` / `gpt.`.
- 2026-10-14: `Gpt.forward_from_emb` split into `trunk_from_emb` (blocks + final LN) + `forward_from_emb` = head(trunk_from_emb) — early fusion needs to stop before the head, exactly as `vlm_engine.py` calls `GPTModel.forward(..., input_embedded=True)`.
- 2026-10-14: Environment — the pinned parity venv lived in `/tmp` and was gone on return; rebuilt from `requirements.txt` + the separate CPU-index torch step (log: `/tmp/venv-build.log`). `pinned.check_pinned()` passes.
- 2026-10-14: **A gate that could not fail, found the same way ViT's V-e was.** V2-j read `activation` off the adapter record and compared it with `Norm.Gelu_exact` — but this test *built* that record, so it compared the test with itself. Mutation 2 made it visible: hardcoding `Gelu_new` inside `adapter_forward` left V2-j green while V2-a/V2-b/V2-e breached at 2.6e-5 / 1.7e-5 against a 1e-6 gate. Rather than reword it, it was deleted — the ffn case of V2-a already measures `ref_vision` against the real `nn.GELU`, so a static gate had nothing left to say. Gate count 31 -> 30.
- 2026-10-14: `test_vlm.ml` compile triad worth remembering: removing the last expression of `run_case` left a trailing `;` before the top-level `let () =`, which the intervening comment hid and OCaml reported only as "Syntax error" at EOF. Same root cause as the usual OCaml-comment quote trap (a lone apostrophe lexing as a char literal) — grep for both when a test file dies at EOF with no line number.
- 2026-10-14: `dune build` was dead before this session and is now green (exit 0). Two causes: `lib/dune` wrapped the library, so every test's `open Tensor` failed as "Unbound module Tensor" — added `(wrapped false)`, which is plainly the intent since the Makefile compiles the same files unwrapped. Then three dead bindings surfaced as warnings-as-errors under dune's dev profile: unused `max_abs_diff` in `test_batched.ml`, unused `vec_max_abs` in `test_mlp.ml`, and an unused `sin_base` from a tuple destructure in `test_rope.ml`. All three removed; `make ci` re-verified green afterwards (199 PASS, exit 0) since those test files changed.
- 2026-10-15: Session closed out at the user's request: **all of D3, D4, D5, ViT, VLM and E1 landed** as one commit `7290c0a` ("ocaml: RoPE, GQA, and the vision half of the quest"), pushed `799052f..7290c0a` to `origin/master`. Re-verified green from the committed tree before pushing: `make ci PARITY_DIR=/tmp/ocaml-parity` exit 0 (42 + 9 + 9 + 16 + 13 + 19 + 13 + 18 + 25 + 30 + 5 gates, gpt2 + kv parity, llama forward E1). Two loose ends closed in the same commit: the four new build products (`test_rope`, `test_gqa`, `test_vit`, `test_vlm`) were untracked and un-ignored, so `.gitignore` gained them; and `ROADMAP.md` still showed D5/E/G as pending plus had no multimodality section at all, so those rows were filled in from the measured gate counts.
- 2026-10-15: Left uncommitted on purpose: `.opencode/GOALS.md` — this file. It is agent-local session state, not part of the port; say the word if it should be tracked or gitignored instead.
- 2026-10-15: **Environment rebuilt from scratch.** The pinned venv had
  lived in `/tmp` and was gone. Rebuilt at `/data/tmp/venv` (build log:
  `/tmp/venv-build.log`); `pinned.check_pinned()` passes. All 12
  references re-exported fp64-current — the previous set in
  `/data/tmp/ocaml-parity` was pre-fp64-calibration F32, so `make parity`
  died on `mat_of_f64: not F64`. Baseline restored: `make ci` 199 PASS.
- 2026-10-15: **`Norm.erf_approx` replaced.** A&S 7.1.26 measured
  2.11e-07 on GELU — the port's one approximation and the stated reason
  six gates sat at 1e-6. Now a Maclaurin series below 2 and an `erfc`
  continued fraction above: **1.11e-15** on erf, 1.33e-15 on GELU, over
  5501 samples on [-12,115] against `math.erf`. All six GELU gates to
  1e-9, measuring 3e-16. Two bugs found by measuring rather than
  reasoning: a Lentz loop that double-inverted `d`, and a series
  recurrence starting at n=1 when term_1 needs n=0's ratio (the second
  was visible as 5.4e-2 at x=0.9975, just below the split).
- 2026-10-15: **V-g retightened** 2.2e-7 → 1e-13. At the old bound it
  measured 2.2e-16 and would have passed on the A&S polynomial swapped
  straight back in — the "gate that cannot fail" shape, found for the
  fourth time. Mutation-verified: the swap makes V-g fail at 2.077e-07
  and breaches all four V-f gates (20 passed/5 failed), then restores to
  25/25.
- 2026-10-15: **Loaders.** Three naming schemes now coexist: HF gpt2
  (`Gpt2_loader`), HF Llama (`Llama_loader`), and the repo's own
  `GPTModel` (`Gpt_repo_loader`, new). The fused checkpoint's `gpt.`
  half needs the third because `vlm_training.py` builds the decoder from
  `llm_quest.gpt.GPTModel`.
- 2026-10-15: **Five facts only a real `state_dict` revealed** — adapter's
  second Linear at `Sequential` index **3** not 2 (the dropout slot sits
  there even as an `Identity`, and nothing in the gap has parameters);
  adapter linears are `bias=False`; `GPTModel.out` is `bias=False` while
  every other Linear is not; `att.mask` is a **bool** buffer (BOOL dtype
  support added — the reader could not open a real file at all); and that
  mask stores `triu(ones, diagonal=1)`, so **1 means masked**, the inverse
  of the port's own `attn_mask`. Also: `assert_exact_keys ~skip_prefix:""`
  filtered out *every* key, since every string starts with "".
- 2026-10-15: **Image preprocessing.** `to_tensor` and ImageNet
  `normalize` bit-identical to torch; `resize_bilinear` at 2.22e-16
  against `interpolate(align_corners=False)`. torchvision was *not*
  installed: 0.29.1 is ABI-incompatible with torch 2.14.1
  (`operator torchvision::nms does not exist`), and installing it risked
  the pin, so the resize gate targets torch's resampler and says so.
- 2026-10-15: **The expensive bug: `input_embedded=True` adds no
  positional embedding.** `GPTModel.forward`'s `x = x + pos_emb` is
  *inside* `if not input_embedded:`. Adding positions measures
  **1.36e-01** (rel ~1e-2) against the reference's logits — enough to
  wreck a caption, far too small for any structural check to notice.
  Found by running gate G-a after the code had been written to match the
  wrong model. Consequence recorded in `gpt.ml`: the reference's trained
  text carries baked-in positions while its *generated* tokens carry
  none. Mirrored, not fixed.
- 2026-10-15: **Two fake gates caught in my own new code** before
  committing: an L5 mask anti-gate that only re-asserted a helper's
  existence (replaced with a real byte-level corruption of the file), and
  an L-h accumulator gate written `near x y && not (near x y)`, which is
  always false. The L-h comment also claimed count-weighting the code did
  not do — the reference averages per-batch means and does *not*
  reweight by example count, so the comment now says that instead.
- 2026-10-15: `make ci` **269 PASS** across 14 suites, exit 0.
- 2026-10-15: **Verifying the docs found two of them lying again, and a
  clean-tree bug that had been invisible.** `make ci` from a *clean* tree
  failed: `test:` listed only up to `test_vlm` as prerequisites, so
  `test_loader_vlm` / `test_image` / `test_loss` were run but never built
  — it had been passing off stale binaries. That is the exact inverse of
  what a CI target is for, and it only shows up after `make clean`.
  Fixed, and `make ci` re-run from scratch: **274 PASS**, exit 0.
- 2026-10-15: **D1's 1.29x margin was the whole of HF's fp32 cast.** The
  RMSNorm gate read 7.767e-07 and the fp32 island (HF's
  `hidden_states.to(torch.float32)`) is 7.767e-07 — to every printed
  digit. So the gate was measuring HF's dtype policy against a
  written-down 1e-6, one BLAS away from a red build that would have
  looked like a port bug. `export_norm.py` now ships a second reference
  (`ref64`, torch's own `functional.rms_norm`, which computes in the
  input dtype), D1a derives its bound from the measured island, and a new
  D1e gates the port against the fp64 kernel at 1e-12: it reads 4.4e-16
  and 2.0e-14. Mutation: rounding the port's output to fp32 **passes**
  D1a and **breaches** D1e at 1.8e-7 — the two gates now demonstrably do
  different jobs, which one gate could not.
- 2026-10-15: **D2 had the same defect with no excuse.** `LlamaMLP` has
  no dtype cast at all — the fp32 reference was purely an export choice —
  so `mlp_ref` gained `ref64` and the same island-derived bound, plus an
  fp64 gate reading 4.4e-16 / 2.2e-15, plus one for the silu grid. The
  comment claiming `LlamaMLP` "casts to float32 before reducing" was
  false and is gone. Same mutation result: fp32-rounded port passes the
  fp32 gate, breaches the fp64 one.
- 2026-10-15: **D3/D4/D5 keep their fp32 references on purpose, and that
  distinction is now written down.** `llm_quest/common/rope.py` asserts
  `dtype == torch.float32` and disables autocast for the angle
  computation; `_compute_llama3_parameters` is fp32; HF's eager
  attention and its `finfo(float32).min` mask are fp32. There, fp32 is
  the implementation's behaviour, so the reference is faithful and the
  gate documents the margin instead of pretending to be tighter. The
  test is "who chose fp32 — the module, or the export?"
- 2026-10-15: **ROADMAP's fp64 numbers were from a different machine.**
  The before/after table said block_0 9.9e-14 / block_11 1.4e-13 /
  logits 1.7e-13; this machine prints 1.4e-13 / 1.7e-13 / 4.4e-13 and
  blocks 1-10 at 1.3e-12..2.3e-12. Replaced with the measured column and
  a note that it is what `make ci` prints here. Same for STUDY.md's
  "every stage to ~1e-13" and the "now measure 3e-16" claim (worst
  reading is 2.7e-15).

- 2026-10-15: **Proofreading pass** (release-readiness audit of `lib/`,
  `test/`, `parity/`). Removed code nothing calls: `Tensor.bmat*` /
  `rand_mat` / `rowvec_mul`, `Block.add_mat` (a duplicate of
  `Tensor.add`) and `block_forward_batch`, `Attention.single_head_no_mask`
  (the repo's v1/v2 teaching form, never used), `Vit.linear_make`
  (byte-identical to `Attention.linear_make`), `Tensor.bmat` type.
  Made gates real: `d5f` now asserts `Attention.gqa_make`'s bias
  structure instead of restating the test's own record; the loss
  NaN-accuracy gate now runs through `accuracy_acc_add`/`_frac`; the
  `hf_eos_contract` check compares the whole greedy prefix, not two
  tautologies; `test_loader`'s C1 wraps the assertion it claimed to
  gate; deleted a rope ref-vs-ref "gate" (both sides were file tensors)
  and its now-dead loads. Fixed the `test_loader_vlm` L5 temp-file leak
  (`Fun.protect`) and a comment that had the corrupted mask columns
  wrong (bytes 4–7, not column 1). Trued up: `tensor.ml` layout header
  (it still said `(d_in x d_out)`), `llama.ml` ("no forward pass yet"),
  `vlm.ml`'s `decode_step_cached` position claim, the `loss.ml` typo,
  `vit.ml`'s "checked rather than assumed", test headers' gate counts,
  `parity.ml`'s ladder (13 hidden states, not 14), README 274 → 273,
  ROADMAP counts (C4 16, D2 22, D4 3/12, M1 20) and the M3 wording
  (the `Vlm.vlm_loss` / `Vit.batch_loss` wrappers are ports, not
  separately gated). Strengthened the vit/vlm references: conv bias is
  no longer all-zero (V-a could not have caught a dropped bias),
  re-exported, suites green. `pinned.check_pinned()` now enforces the
  `safetensors` pin too. `Config.gpt_small_config` now uses exact
  `Gelu`: the repo's own `gpt_transformer_block.GELU` is the erf form,
  and `GPT_SMALL_CONFIG` does carry `qkv_bias: False` after all.
- 2026-10-15: Docs structured for mobile reading: a "Where to start"
  path at the top of `ocaml/README.md` (README → ROADMAP → STUDY →
  code in dependency order → `make ci`), and a pointer block on the
  workspace root README so the port is reachable from the repo landing
  page. Verified from a clean tree: `make clean && make ci` exit 0,
  273 PASS, 0 FAIL.
