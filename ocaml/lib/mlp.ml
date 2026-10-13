(** SwiGLU feed-forward — the Llama FFN, mirroring [LlamaMLP]
    (transformers) and [SiLU]/[FFN] in
    [gpt_to_llama3/llama_transformer_block.py].

    {[
      FFN_SwiGLU(x) = (silu(x @ gate_proj^T) * (x @ up_proj^T)) @ down_proj^T
    ]}

    Three [(out, in)] weights — [gate_proj] [intermediate, hidden],
    [up_proj] [intermediate, hidden], [down_proj] [hidden,
    intermediate] — and **no biases**. Llama's linears are bias-free
    (see [Config.qkv_bias]), so the record carries no [b] and the
    kernel takes none: a bias cannot be added at a call site, it has
    to be added here, and gate D2b pins that. GPT-2's FFN in
    [Block] has biases on both projections and a pointwise GELU
    between them; the same three structural facts are what D2b
    differentiates on.

    The activation choice is not a parameter. [silu] *is* SwiGLU —
    "silu on the gate, no flag" is the name's contract, the same
    convention as [Norm.rms_norm] — and gate D2e pins that using
    GELU-on-the-gate produces a different function. *)

open Tensor

(** The gate/up/down triple of a Llama FFN layer. *)
type swiglu = {
  gate_proj : mat; (** [intermediate, hidden] *)
  up_proj : mat; (** [intermediate, hidden] *)
  down_proj : mat; (** [hidden, intermediate] *)
}

(** Numerically stable sigmoid. The naive [1 / (1 + exp(-x))]
    overflows fp64 for [x < -709], where the true sigmoid is still
    [~1e-308] — the sigmoid collapses to 0 *before* the kernel has
    said what silu returns there — and the reciprocal form
    [exp(x) / (1 + exp(x))] is inf/inf = [nan] for [x > 709]. Both
    branches below keep the overflowing exponential out of the
    denominator. *)
let sigmoid x =
  if x >= 0.0 then 1.0 /. (1.0 +. exp (-. x))
  else
    let e = exp x in
    e /. (1.0 +. e)

(** silu(x) = x * sigmoid(x). *)
let silu_scalar x = x *. sigmoid x

let silu_vec (v : vec) : vec = Array.map silu_scalar v

let silu_mat (m : mat) : mat = Array.map silu_vec m

(** The SwiGLU forward. Accumulates in OCaml's [float] (fp64)
    rather than the fp32 [LlamaMLP] casts to. That is strictly more
    accumulation precision, not a different function — the difference
    is ~1e-6 relative at these magnitudes, which is what D2a's dual
    abs+rel gate is calibrated against. *)
let forward (m : swiglu) (x : mat) : mat =
  let g = linear_fwd m.gate_proj x None in
  let u = linear_fwd m.up_proj x None in
  linear_fwd m.down_proj (hadamard (silu_mat g) u) None
