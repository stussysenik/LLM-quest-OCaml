(** LayerNorm + GELU, mirroring [llm_quest/gpt/gpt_transformer_block.py].

    - [layernorm]: affine Z-score over the last dim, per token:
      [y = scale * ((x - mean) / (std + eps)) + shift].
      Matches the repo Python (eps added outside the sqrt,
      [unbiased=false]). Set [eps_inside_sqrt] for HuggingFace/PyTorch
      LayerNorm convention ([(x-mean)/sqrt(var+eps)]): the two differ
      at ~5e-6 relative — invisible in training, fatal for a 1e-5
      parity gate over 12 layers. The HF weight loader sets [true].
    - [gelu]: exact erf version [x * 0.5 * (1 + erf(x / sqrt 2))].
      OCaml stdlib has no [erf], so we use the Abramowitz & Stegun
      7.1.26 approximation (max err ~1.5e-7). *)

open Tensor

type layernorm = { scale : vec; shift : vec; eps : float; eps_inside_sqrt : bool }

let layernorm_make dim : layernorm =
  { scale = ones_vec dim; shift = zeros_vec dim; eps = 1e-5; eps_inside_sqrt = false }

let layernorm_forward (ln : layernorm) (x : mat) : mat =
  Array.map
    (fun row ->
      let mu = mean_vec row in
      let var = ref 0.0 in
      Array.iter (fun v -> var := !var +. ((v -. mu) *. (v -. mu))) row;
      var := !var /. float_of_int (Array.length row);
      let denom =
        if ln.eps_inside_sqrt then sqrt (!var +. ln.eps) else sqrt !var +. ln.eps
      in
      Array.mapi (fun j v -> ln.scale.(j) *. ((v -. mu) /. denom) +. ln.shift.(j)) row)
    x

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
