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

    - [gelu]: exact erf version [x * 0.5 * (1 + erf(x / sqrt 2))].
      OCaml stdlib has no [erf], so we use the Abramowitz & Stegun
      7.1.26 approximation (max err ~1.5e-7). *)

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

(** erf approximation (Abramowitz & Stegun 7.1.26). *)
let erf_approx x =
  let a1 = 0.254829592 in
  let a2 = -0.284496736 in
  let a3 = 1.421413741 in
  let a4 = -1.453152027 in
  let a5 = 1.061405429 in
  let p = 0.3275911 in
  let sign = if x < 0.0 then -1.0 else 1.0 in
  let ax = abs_float x in
  let t = 1.0 /. (1.0 +. p *. ax) in
  let y =
    1.0 -. (((((a5 *. t +. a4) *. t +. a3) *. t +. a2) *. t +. a1) *. t *. exp (-.ax *. ax))
  in
  sign *. y

let gelu_scalar x = x *. 0.5 *. (1.0 +. erf_approx (x /. sqrt 2.0))

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
