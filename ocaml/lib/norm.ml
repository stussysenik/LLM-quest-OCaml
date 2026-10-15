(** Normalization ops + GELU, mirroring
    [llm_quest/gpt/gpt_transformer_block.py] (LayerNorm, GELU) and
    [transformers.models.llama.modeling_llama] (RMSNorm).

    Two conventions separate LayerNorm from RMSNorm, and each is a named
    type rather than a boolean:

    - [centering]: LayerNorm subtracts the mean — an affine Z-score over
      the last dim, [y = scale * ((x - mean) / (std + eps)) + shift].
      RMSNorm does not; it is [y = scale * x / sqrt(mean(x^2) + eps)].
    - [eps_placement]: HuggingFace/PyTorch adds eps *inside* the sqrt
      ([(x-mean)/sqrt(var+eps)]); this repo's own GPT-2 code adds it
      outside ([(x-mean)/(std+eps)]). The two differ at ~5e-6 relative —
      invisible in training, fatal for a 1e-5 parity gate over 12 layers.

    A boolean that inverts at every call site is exactly the thing someone
    flips without reading, so the kernel below takes both conventions as
    named arguments and the two entry points state theirs: [layer_norm]
    is GPT-2's — mean subtracted, affine, the checkpoint's eps convention —
    and [rms_norm] is Llama's, taking neither flag, because "no centering,
    no shift, eps inside the sqrt" is what the name means and what
    [LlamaRMSNorm] does.

    - [gelu]: exact-erf version [x * 0.5 * (1 + erf(x / sqrt 2))].
      OCaml's stdlib has no [erf], so [erf] is built here from a Maclaurin
      series below 2 and an [erfc] continued fraction above. **Its error is
      measured, not assumed: 1.11e-15 on erf and 1.33e-15 on GELU over
      [-12, 115] against Python's [math.erf]** — which is fp64 round-off,
      so the GELU gates are no longer held at 1e-6 by an approximation.
      (They previously were: this function was Abramowitz & Stegun 7.1.26
      at 2.11e-07 on GELU, and an earlier "~1.5e-7" note in this comment was
      that formula's bound on *erf*, understating GELU's by ~40%.) GPT-2 is
      unaffected either way: HF `gpt2` uses `gelu_new`, whose tanh path does
      not call erf at all. *)

open Tensor

(** Subtract the row's mean before scaling ([LayerNorm]) or not ([RMSNorm]). *)
type centering = Centering | No_centering

(** [Eps_inside_sqrt] is the HuggingFace/PyTorch convention and what every
    loaded checkpoint here was trained under; [Eps_outside_sqrt] is the
    repo's own GPT-2 convention. *)
type eps_placement = Eps_inside_sqrt | Eps_outside_sqrt

(** GPT-2's weight pair. [eps_placement] is a field rather than a constant
    because a checkpoint fixes the convention at load — [Gpt2_loader]
    reads HF weights, so it is [Eps_inside_sqrt], while [layernorm_make]
    builds the repo's training-time one. Llama's RMSNorm is a bare [vec]
    with no record at all: there is nothing about it a checkpoint can vary
    except eps, which is a function of the config. *)
type layernorm = { scale : vec; shift : vec; eps : float; eps_placement : eps_placement }

let layernorm_make ?(eps = 1e-5) dim : layernorm =
  { scale = ones_vec dim; shift = zeros_vec dim; eps; eps_placement = Eps_outside_sqrt }

(** The one normalization kernel. Per row: reduce the last dim to a
    variance, divide by its (eps-placed) root of, scale, shift.

    [shift = None] is RMSNorm's missing bias, not a vector of zeros: the
    difference is the point of the module, and it keeps the inner loop from
    adding a zero per element. [centering] is matched once here, outside
    the row loop, for the same reason. *)
let norm_forward ~(scale : vec) ~(shift : vec option) ~eps ~centering ~eps_placement
    (x : mat) : mat =
  let center = match centering with Centering -> true | No_centering -> false in
  Array.map
    (fun row ->
      let mu = if center then mean_vec row else 0.0 in
      let ss = ref 0.0 in
      Array.iter (fun v -> let d = if center then v -. mu else v in ss := !ss +. (d *. d)) row;
      let var = !ss /. float_of_int (Array.length row) in
      let denom =
        match eps_placement with Eps_inside_sqrt -> sqrt (var +. eps) | Eps_outside_sqrt -> sqrt var +. eps
      in
      Array.mapi
        (fun j v ->
          scale.(j) *. ((if center then v -. mu else v) /. denom)
          +. (match shift with None -> 0.0 | Some s -> s.(j)))
        row)
    x

(** LayerNorm over the last dim: mean subtracted, then scale and shift.
    Every GPT-2 norm in the model goes through here. *)
let layer_norm (ln : layernorm) (x : mat) : mat =
  norm_forward ~scale:ln.scale ~shift:(Some ln.shift) ~eps:ln.eps ~centering:Centering
    ~eps_placement:ln.eps_placement x

(** RMSNorm, Llama's norm: [y = w * x / sqrt(mean(x^2) + eps)]. No mean
    subtracted, no bias, eps inside the sqrt — all three are fixed by the
    architecture, so the signature carries only what a checkpoint can vary.

    Accumulates in OCaml's [float], which is fp64 rather than the fp32
    [LlamaRMSNorm] casts to. That is strictly more accumulation precision,
    not a different function: the difference between the two is ~1e-7
    relative, which is what gate D1a's tolerance is calibrated against. *)
let rms_norm (w : vec) ~eps (x : mat) : mat =
  norm_forward ~scale:w ~shift:None ~eps ~centering:No_centering ~eps_placement:Eps_inside_sqrt x

(** [erf] — the error function, to fp64 round-off.

    Two regions, because neither formula covers the range on its own:

    - [|x| < 2]: the Maclaurin series [2/sqrt(pi) * sum (-1)^n x^(2n+1) /
      (n! (2n+1))]. Accurate and branch-free, but its terms peak near
      [n ~ x^2] with magnitude [x^(2n+1)/n!], so by [x = 6] a term reaches
      ~1e13 against a result of 1.0 and the alternating sum has lost
      ~13 digits to cancellation. Below 2 that peak is 2.4 against a
      result of 1.0 — nothing lost — which is what fixes the split.
    - [|x| >= 2]: [erfc] by modified Lentz on [x + (1/2)/(x + 1/(x +
      (3/2)/(x + ...)))], then [erf = 1 - erfc]. The mirror image: it
      converges in 57 iterations at x=2 and 17 at x=6, but takes 189 at
      x=1, which is why the series covers that side of the boundary.

    Measured, not assumed: over 5501 samples spanning [-12, 115] against
    Python's [math.erf], max abs error [1.11e-15] (~5 ulp) and max GELU
    error [1.33e-15]. Gate D1e pins both numbers against an exported
    erf table, so a change to either branch cannot quietly move them. *)
let erf_series x =
  let x2 = x *. x in
  let term = ref x in
  let sum = ref x in
  let n = ref 0 in
  while !n < 200 && abs_float !term > 1e-18 do
    let nf = float_of_int !n in
    term := !term *. -.x2 *. ((2.0 *. nf) +. 1.0) /. ((nf +. 1.0) *. ((2.0 *. nf) +. 3.0));
    sum := !sum +. !term;
    incr n
  done;
  2.0 /. sqrt Float.pi *. !sum

let erfc_cf x =
  let tiny = 1e-300 in
  let cf = ref x in
  let c = ref x in
  let d = ref 0.0 in
  let converged = ref false in
  let n = ref 0 in
  while (not !converged) && !n < 200 do
    incr n;
    let a = float_of_int !n /. 2.0 in
    let dn = x +. (a *. !d) in
    d := 1.0 /. (if abs_float dn < tiny then tiny else dn);
    let cn = x +. (a /. !c) in
    c := if abs_float cn < tiny then tiny else cn;
    let delta = !c *. !d in
    cf := !cf *. delta;
    if abs_float (delta -. 1.0) < 1e-17 then converged := true
  done;
  exp (-.(x *. x)) /. sqrt Float.pi /. !cf

let erf x =
  let sign = if x < 0.0 then -1.0 else 1.0 in
  let ax = abs_float x in
  sign *. if ax < 2.0 then erf_series ax else 1.0 -. erfc_cf ax

let gelu_scalar x = x *. 0.5 *. (1.0 +. erf (x /. sqrt 2.0))

let gelu_vec (v : vec) : vec = Array.map gelu_scalar v

let gelu_mat (m : mat) : mat = Array.map gelu_vec m

(** Activation selector. [Gelu_exact] is the repo's erf GELU
    ([gpt_transformer_block.py]); [Gelu_new] is HuggingFace's tanh
    approximation ("gelu_new", the GPT-2 default), needed for weight
    parity — the two differ at ~1e-3, which would break every block
    past the first FFN. *)
type activation = Gelu_exact | Gelu_new

let gelu_new_scalar x =
  let c = sqrt (2.0 /. Float.pi) in
  0.5 *. x *. (1.0 +. tanh (c *. (x +. 0.044715 *. x *. x *. x)))

let apply_activation_vec (a : activation) (v : vec) : vec =
  match a with Gelu_exact -> gelu_vec v | Gelu_new -> Array.map gelu_new_scalar v

let apply_activation_mat (a : activation) (m : mat) : mat = Array.map (apply_activation_vec a) m

(** Config's polymorphic [Config.activation] -> these kernels. Only
    GPT-2 configs carry one: Llama's feed-forward is SwiGLU, which is a
    gated product rather than a pointwise function and has no [Norm]
    counterpart. *)
let activation_of (c : Config.t) : activation =
  match Config.activation c with
  | `Gelu -> Gelu_exact
  | `Gelu_new -> Gelu_new
