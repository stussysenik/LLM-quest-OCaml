# Study guide — what transfers from this port

Not a tutorial. A list of the decisions that were expensive here, written
so the reasoning can be reused on a different codebase, with the failure
that motivated each one attached so the rule can be told from a
preference.

Read this if you are about to: port a model between frameworks, write
parity tests, set a numerical tolerance, or review code that claims a test
proves something.

---

## 1. Calibrate the tolerance against the reference, not against yourself

**The rule.** Before choosing a gate, find out how much *the reference
disagrees with itself*. If PyTorch disagrees with its own fp64 run by
2.3e-6 on your machine, then no fp32 comparison of yours can be gated
tighter than that, and a 1e-6 gate is testing the host's BLAS reduction
order.

**What happened.** A faithful OCaml GPT-2 port breached at rel ~1e-6
against an fp32 PyTorch reference. The port looked broken. It wasn't:
PyTorch disagreed with *its own* fp64 run by 2.3e-6 at block 3 — the
reference's own fp32 rounding exceeded the gate, and the number moved
with the machine.

**The fix.** Export reference activations in **float64**, keep weights in
**F32** as checkpointed. Every stage dropped from ~1e-6 to between
**0.0** (the embedding lookup, bit-identical) and **2.3e-12** (the
deepest blocks; the final logits read 4.4e-13) and the number became
platform-independent. The gate went from 1e-6 to **1e-9**, seven orders
tighter.

**Why fp64 for the reference and F32 for the weights.** Weights are F32
because that is what a checkpoint *is*; activations are F64 because the
gate compares two implementations of one function and the tolerance it
can hold is set by the less precise of the two.

**Reusable form.** Before picking a number, run the reference against
itself at higher precision. If it wobbles, your gate is measuring the
reference, not the port.

**The same defect at a smaller scale, and the better fix.** This section
was written for the full-model stages. The per-operator gates (RMSNorm,
SwiGLU) had the identical problem and hid it behind a written-down
constant. `LlamaRMSNorm.forward` hard-casts `hidden_states.to(torch.float32)`,
so its output carries an fp32 *island* — and that island was worth
7.767e-07 on the mid case, which was, **to every printed digit**, what the
gate reported. The gate was measuring HF's dtype policy at a 1.29x margin
against a 1e-6 bound: one BLAS reduction order away from a red build that
would have looked like a port bug.

Two changes, and the second is the transferable one:

1. **Derive the bound instead of writing it down.** Both references ship
   (`.ref` fp32, `.ref64` fp64), the gate measures the gap between them in
   the test, and asserts `|port - ref| <= 2 x island`. The bound now
   tracks whatever the reference's precision costs on *this* machine.
   `LlamaMLP` had the same issue with no such excuse — its `forward` has no
   cast at all, so its fp32 reference was purely an export choice.

2. **Separate "is it faithful" from "is it accurate".** These are two
   claims and one gate cannot hold both when the reference is imprecise.
   Now: gate A vs the module as it actually runs (HF's cast included), and
   gate B vs the same arithmetic in fp64 — for RMSNorm, torch's
   *own* `functional.rms_norm` kernel, so gate B is not the port's
   formula echoed back at it.

   The payoff is a single mutation that shows exactly what each gate
   knows: round the port's output to fp32 and **gate A passes** (an fp32
   value is indistinguishable from HF's fp32 module — gate A never could
   have caught this) while **gate B breaches** at 1.8e-07 against 1e-12.
   See `test_norm.ml` / `test_mlp.ml`, and ROADMAP's D1/D2 rows.

   Not every fp32 reference is a defect, though: `llm_quest/common/rope.py`
   *asserts* `dtype == torch.float32` and disables autocast, and
   `_compute_llama3_parameters` is fp32 — there, fp32 is the
   implementation's behaviour and the reference is faithful. The
   test is "who chose fp32, the module or the export?"

---

## 2. A gate you have never seen fail is an assumption

**The rule.** Every gate set gets at least one deliberate mutation, and
the mutation must be *observed* to fail. Not reasoned about. Run.

**What happened.** This repo found five gates that could not fail:

| Gate | Why it couldn't |
| :--- | :--- |
| D3b | compared two unrelated vectors; nothing tied them to the op |
| ViT V-e | compared the test's *own* record against a hardcoded constant |
| VLM V2-j | compared a record the test itself constructed — the test built it, so it agreed with itself by construction |
| V-g | bound at 2.2e-7 while measuring 2.2e-16, so swapping back to the approximation it replaced passed |
| L-h | `near x y && not (near x y)` — always false |

Four were found *by running a mutation*. The V-g one was found by noticing
the gate was nine orders looser than the thing it measured.

**The meta-lesson.** V-g had been fine, and became a fake gate the moment
the code it measured got better. **A tolerance gate is only as tight as
the worst thing it is willing to catch**, and tightening code without
re-tightening its gate is how a gate quietly dies.

**Reusable form.** For each gate: *name the bug it catches*. If you can't,
delete it — it is costing maintenance and buying nothing.

---

## 3. Anti-gates: where the correct implementation must *disagree*

**The rule.** Some properties are better pinned by a negative than a
positive. When an operation is a permutation, a re-ordering, or a
choice between conventions, assert that the *wrong* one provably differs
— not merely that the right one matches.

**Examples from this repo:**

- **Grouped-query attention.** GQA is a permutation of heads, so a
  repeat-tiling implementation can look plausible. Gate D5c asserts the
  tile implementation **must disagree** (>1e-3) and that the correct one
  is the closer match. It is the only gate that catches an
  implementation that is accidentally right.
- **Early fusion.** Asserting "the image embeddings come first" only
  proves the code says so. The anti-gate asserts the *swapped* order
  diverges.
- **Image resize.** On a 9×13 input, an axis-swapped resize and a
  half-pixel-off resampler both produce 16×16 output. Every shape
  assertion passes. Only a divergence check catches them.

**Reusable form.** Ask "what is the *near miss* here?" Then assert the
near miss fails.

---

## 4. Square matrices hide transposes

**The rule.** A shape assertion cannot catch a transposed weight when the
matrix is square. Add a gate that re-derives the expected dimension from
the *config*, independently of the file.

**Where it bit.** Every attention projection is `n_heads × head_dim` wide
when `hidden = n_heads × head_dim`, which is true of essentially every
real model. `q_proj`, `k_proj`, `v_proj`, `o_proj` are all square on the
tiny test configs *and* on ViT-base. A loader that transposed one loads
without error and produces wrong numbers whose only symptom is parity.

**The defence, used throughout:** `check_layout` re-derives each matrix's
expected `out` axis from the config and compares it to the loaded rows —
independently of the `~expected_shape` check, which only says "the file
has the shape the config implies" and is satisfied by a loader that
transposed *every* matrix consistently.

**Reusable form.** For each weight, ask: *would the wrong version have
the same shape?* If yes, shape checks are not verification.

---

## 5. Order-sensitive ops need anti-gates, not tolerances

**The rule.** Concatenation order, patch traversal order, and slice
offsets produce self-consistent models. A wrong order is not a small
error — it is a *different model* that matches nothing.

**Where it bit.**

- ViT patch order is row-major over the `(H/P, W/P)` grid, and the
  positional table is indexed against it. A column-major mix-up yields a
  model that learns to compensate. Gate V-c pins it.
- `vlm_loss` slices `logits[:, nv-1:-1]`. Off-by-one here drops a real
  target or invents one, and the loss still looks reasonable.
- `GPTModel.out` is bias-free while every other `Linear` in it is not.

**Reusable form.** For any slice, offset or concatenation, ask: *if this
were off by one, would the output still look like a number?* If yes, gate
it with a divergence check against the known-wrong variant.

---

## 6. Export a real artifact, not a hand-written table

**The rule.** Any key you *believe* a checkpoint has should be checked
against a `state_dict` you actually called. Beliefs about serialisation
formats are wrong in specific, learnable ways.

**Five bugs found this way in one pass**, none of which reading the
loader would have found:

| Believed | Actually |
| :--- | :--- |
| adapter's second Linear is `Sequential` index 2 | index **3** — `Sequential(Linear, GELU, Dropout\|Identity, Linear)`; no module in the gap has parameters, so the checkpoint shows 0 and 3 with nothing between |
| the adapter's Linear has biases | `bias=False`; no checkpoint has them |
| the LM head has a bias | `GPTModel.out` is `bias=False` while every other Linear is not |
| the attention mask is a float tensor | a persistent **bool** buffer — the reader could not open a real file at all |
| mask `1` means attend | `triu(ones, diagonal=1)` — **1 means masked**, the inverse of the library's own convention |

**Reusable form.** Before writing a loader or a parser, make the other
side dump its real structure and read it. Ten minutes against an hour of
being wrong.

---

## 7. Polarity conventions deserve their own gate

**The rule.** When two components in a pipeline disagree about what a
boolean means, say which one is right *in the gate*, do not inherit it.

**Where it bit.** The repo's `att.mask` stores 1 = masked. The port's
`attn_mask` is a bool array where `true` = attend. Read the buffer's
booleans straight into the library and you get a **bidirectional
decoder** that generates fluent nonsense, trains, and fails no shape
check anywhere. `check_mask` now states the polarity explicitly
(`c > r → masked`) rather than deriving it from either side.

**Reusable form.** Grep for every boolean that crosses a module boundary.
For each, write down what `true` means on both sides and check they match.

---

## 8. Look for the branch that skips the whole thing

**The rule.** Read the code you are porting for *absence*, not presence.
A flag that disables an entire stage is easy to miss because the
surrounding code looks like it still runs.

**Where it bit — the most expensive bug in the port.**
`GPTModel.forward` contains `x = x + pos_emb`. It looks unconditional. It
is inside `if not input_embedded:`. So the multimodal path, which passes
`input_embedded=True`, gets **no positional embedding at all** — on the
image and on every generated token.

Adding positions (the natural assumption, and what `Gpt.add_pos` exists
for and is correct for everywhere else) measures **1.36e-01** against the
reference's logits: rel ~1e-2, enough to wreck a caption, and far too
small for a shape assertion, a key-set check or a range check to notice.

It was found by a gate, after the code had already been written to match
the wrong model.

**Reusable form.** For every `if` in the reference that gates a whole
operation, ask which of your code paths lands on the other side.

---

## 9. Measure tolerances; do not inherit them

**The rule.** When a comment states a bound, re-measure it. Bounds are
written once and then outlive the code that justified them.

**What happened.** `Norm.erf_approx` (Abramowitz & Stegun 7.1.26) had a
comment claiming "~1.5e-7". Measured: **2.11e-07 on GELU** over
[-12, 12] — the stated number was the formula's own bound on *erf*, and
GELU's error was ~40% worse. That 2.11e-07 was why six gates sat at 1e-6
instead of 1e-9.

**The fix.** Replaced with a Maclaurin series below 2 and an `erfc`
continued fraction above: **1.11e-15** on erf, **1.33e-15** on GELU,
against Python's `math.erf` over 5501 samples on [-12, 115]. All six gates
moved to 1e-9. Their worst reading in `make ci` is now 2.7e-15
(`V-f hidden states`, the ViT's deepest margin); the GELU gate itself
reads 2.2e-16.

**Reusable form.** Every numeric bound in a comment is a claim. Re-derive
it, and when it turns out to be load-bearing (that 2.11e-07 was holding
six tolerances down), replace the thing rather than relaxing the gate.

---

## 10. Near-misses are the bugs worth hunting

**The rule.** The most valuable bug you will find is one that produces a
*plausible wrong answer* rather than an error. Those are found by asking
what a plausible-looking mistake would produce and checking the number.

**Worked example — the ViT.** The obvious errors all fail loudly:
wrong patch size, wrong image dimensions, wrong class count. The
interesting one is that `nn.Conv2d` is the only weight in the project
outside the `[out, in]` convention, and flattening it to a matrix at load
is *arithmetically fine* — because `emb_dim` and `num_channels * P * P`
are both **768** for ViT-base. So a loader could reshape it, destroy the
gate that notices, and pass everything. The layout is therefore kept as
`vec array` with the axes flattened *only* at the point of use, and the
gate reads the library default rather than a constant.

**Reusable form.** List the changes that would still compile. Those are
your real bug surface.

---

## 11. Pin the environment where it can bite

**The rule.** `pinned.check_pinned()` runs at import in every export
script and raises on any mismatch. A silently-upgraded dependency turns a
phantom regression into an afternoon.

**Two environment bugs worth knowing:**

- **The SIGBUS.** HF loads safetensors via `mmap`, and gpt2's tied
  `lm_head.weight` came back with `data_ptr() % 64 == 19`. torch's CPU
  BLAS picks a vectorised GEMV path for single-token forwards that needs
  64-byte-aligned weights, and dereferences it anyway — SIGBUS, exit 138,
  no Python traceback. `seq_len >= 2` (GEMM) was fine, so every earlier
  export succeeded and only the single-token ones died. Fix:
  `align_parameters()` clones params into aligned storage, bit-identical
  values. **A crash with no traceback, only in one code path, is an
  alignment or a lifetime bug before it is anything else.**
- **The CPU index.** `--index-url` inside a requirements file is a
  *global* option, so putting the CPU index there sends `transformers` and
  `safetensors` to `download.pytorch.org` too, where they do not exist;
  leaving it out lets PyPI serve the CUDA build. A CUDA torch on a CPU box
  and a CPU torch run different kernels in different reduction orders, so
  a reference exported on one and gated against the other is a
  coincidence, not a comparison. torch is installed in a separate step
  from the CPU index only, and `pinned.py` refuses to export unless the
  version is the pinned CPU build.

---

## 12. Review the gate, not the diff

**The rule.** When you change code that gates have, the gate set is part
of the change. Re-read the tolerances in the same sitting as the
implementation.

**Worked example.** Six gates sat at 1e-6 with a comment explaining why.
The reason evaporated. Nobody re-read the comment, so the gates stayed at
1e-6 — an order of magnitude looser than the code allowed, and looser
than the errors they were catching (2.11e-07) by a small margin, so they
still passed and nobody noticed they had stopped protecting anything.

**Reusable form.** Grep the tolerances. For each, find the sentence that
justifies it. If the justification has changed, the number is wrong.

---

## Where to start reading the code

| If you want to see… | Read |
| :--- | :--- |
| the layout convention and why it is `[out, in]` | `lib/attention.ml`, then step L in `ROADMAP.md` |
| a parity harness that erases a family behind a record | `test/parity.ml` |
| how a tolerance was chosen | `lib/norm.ml`'s header, and `ROADMAP.md`'s fp64 story |
| a gate that catches a near-miss | `test/test_gqa.ml` (D5c), `test/test_vit.ml` (V-c), `test/test_image.ml` (I-d) |
| a loader that had to be rewritten by reality | `lib/gpt_repo_loader.ml`'s header |
| the most expensive single bug | `lib/gpt.ml`, `prefill_from_emb_cached` |

And the rule that produced all of it, from the review that set the
original order:

> **Verification before optimization.** Each step must be numerically
> verified against PyTorch before the next step begins. Never debug two
> things at once ("is my batching correct?" + "is my architecture
> correct?").
