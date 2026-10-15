(** RMSNorm gates D1a-D1d (ROADMAP step D1).

    Usage: [./test_norm <norm_ref.safetensors>]
    (written by [ocaml/parity/export_norm.py], which instantiates
    [LlamaRMSNorm] rather than restating its formula).

    RMSNorm is the first Llama delta:
    [y = w * x / sqrt(mean(x^2, dim=-1) + eps)] per token, accumulated in
    fp32 regardless of input dtype. It is also the only delta that is not a
    composition, so nothing here touches a model — the op is proved in
    isolation and first composed in step E.

    | D1a | vs [LlamaRMSNorm] at hidden=16 and hidden=2048, 1e-6 |
    | D1b | differential vs GPT-2 LayerNorm: identical once the input's
           mean is zero, and differing in the three named ways after that |
    | D1c | the eps-placement convention changes the result |
    | D1d | [Config.norm_eps] is family-specific, and is what D1a ran with |

    D1a compares against the file's own inputs, so it cannot pass by
    agreeing with PyTorch on the wrong numbers. D1b is the gate that earns
    its keep: an RMSNorm that kept the mean subtraction differs from
    [LlamaRMSNorm] on most inputs, but *passes* on inputs whose mean is
    ~0 — so the cheap way to catch "accidentally implemented LayerNorm" is
    not the comparison against the reference, it is the algebraic
    identity. On a zero-mean row the two ops are the same function, which
    proves the mean subtraction is the *only* difference; shift sensitivity
    proves that difference is real. *)

open Tensor

let passed = ref 0
let failed = ref 0

let check name ok =
  if ok then (
    incr passed;
    Printf.printf "PASS %s\n" name)
  else (
    incr failed;
    Printf.printf "FAIL %s\n" name)

(* ------------------------------------------------------------------ *)
(* Numeric helpers — the same as [parity.ml]'s, so that a number printed
   here means the same thing as one printed there.                       *)
(* ------------------------------------------------------------------ *)

let mat_max_abs (m : Tensor.mat) : float =
  let v = ref 0.0 in
  Array.iter (Array.iter (fun x -> v := max !v (abs_float x))) m;
  !v

let max_abs_diff_mat (a : Tensor.mat) (b : Tensor.mat) : float =
  let rows = Array.length a in
  if rows <> Array.length b then failwith "test_norm: row count mismatch";
  let m = ref 0.0 in
  for i = 0 to rows - 1 do
    if Array.length a.(i) <> Array.length b.(i) then failwith "test_norm: col count mismatch";
    for j = 0 to Array.length a.(i) - 1 do
      m := max !m (abs_float (a.(i).(j) -. b.(i).(j)))
    done
  done;
  !m

(** The numeric stage line, in [parity.ml]'s format and with its dual
    abs+rel criterion. The dual gate is not decoration here: the reference
    is fp32 — [LlamaRMSNorm] casts to float32 before reducing — so at
    hidden=2048, where |y| reaches ~10, one fp32 ulp is already ~1e-6
    absolute, and a flat absolute gate would be measuring the reference's
    own rounding rather than our error. The relative criterion is what
    pins the computation; the absolute one still catches a whole-tensor
    shift that relative error hides at small magnitudes. *)
let gate name ours refm abs_tol rel_tol =
  let diff = max_abs_diff_mat ours refm in
  let refmag = mat_max_abs refm in
  let rel = diff /. max refmag 1e-30 in
  let ok = diff <= abs_tol && rel <= rel_tol in
  if ok then incr passed else incr failed;
  Printf.printf "%-14s max_abs_diff=%.3e ref_max=%.3e rel=%.3e abs_tol=%.0e rel_tol=%.0e %s\n" name
    diff refmag rel abs_tol rel_tol
    (if ok then "OK" else "BREACH")

(** "These must NOT agree", with the size of the disagreement reported.
    [min_gap] is the point of the test: fp64 noise on a value of order 1 is
    ~1e-16, so a threshold of 1e-9 is "structurally different" while still
    leaving room for the difference being genuinely small. *)
let check_diverges name ~min_gap a b =
  let d = max_abs_diff_mat a b in
  let ok = d > min_gap in
  check (Printf.sprintf "%s (max_abs_diff=%.3e, must exceed %.0e)" name d min_gap) ok

let () =
  if Array.length Sys.argv <> 2 then (
    Printf.eprintf "usage: %s <norm_ref.safetensors>\n" Sys.argv.(0);
    exit 2);
  let f = Safetensors.open_file Sys.argv.(1) in

  let load_case name =
    let x = Safetensors.mat_of f (Printf.sprintf "d1.%s.x" name) in
    let w = Safetensors.vec_of f (Printf.sprintf "d1.%s.w" name) in
    let eps = (Safetensors.vec_of f (Printf.sprintf "d1.%s.eps" name)).(0) in
    let ref = Safetensors.mat_of f (Printf.sprintf "d1.%s.ref" name) in
    let ref64 = Safetensors.mat_of_f64 f (Printf.sprintf "d1.%s.ref64" name) in
    if Array.length x = 0 || Array.length w <> Array.length x.(0) then
      failwith
        (Printf.sprintf "test_norm: case %s: w has %d entries, x's last dim is %d" name
           (Array.length w)
           (if Array.length x = 0 then 0 else Array.length x.(0)));
    (x, w, eps, ref, ref64)
  in

  (* eps is [Config.norm_eps] of the tiny-random Llama config, so gate D1a
     runs the op through the accessor rather than a constant — and D1d
     below checks the wiring that makes that true. *)
  let cfg = Config.llama_tiny_random_config in
  let eps_cfg = Config.norm_eps cfg in

  (* ---------------- D1a: vs LlamaRMSNorm ---------------- *)
  (* The reference is the module HF actually runs, unpatched — including
     its `hidden_states.to(torch.float32)`. That cast is not a detail of
     the gate's setup, it is what the module does, so this gate's job is
     "does the port track that module", not "is the port accurate": the
     accuracy claim is D1e's, against torch's fp64 kernel.

     The bound is therefore derived rather than written down. `island` is
     measured here from the two references in the file, and the port has
     to land within 2x of it — |port - ref| <= |port - ref64| + island,
     so 2x is a real margin over anything the port can contribute. On
     this machine the island *is* the entire error D1a reports, which is
     how it was identified: a 1e-6 constant gave a 1.29x margin that
     would have gone red on a host with a different BLAS reduction
     order. See STUDY.md §1. *)
  Printf.printf "D1a: rms_norm vs LlamaRMSNorm (eps=%.9g from Config.norm_eps)\n" eps_cfg;
  Printf.printf
    "D1e: rms_norm vs torch.nn.functional.rms_norm (fp64 kernel, no dtype cast)\n";
  let run_case label =
    let x, w, eps_ref, ref, ref64 = load_case label in
    let ours = Norm.rms_norm w ~eps:eps_cfg x in
    let island = max_abs_diff_mat ref ref64 in
    let bound = Float.max (2.0 *. island) 1e-12 in
    gate label ours ref bound bound;
    Printf.printf
      "    hidden=%d eps_in_file=%.9g |ref|max=%.3e fp32-island=%.3e bound=%.3e\n"
      (Array.length w) eps_ref (mat_max_abs ref) island bound;
    (* The port is fp64 throughout and torch's kernel computes in the
       input dtype, so with the fp32 cast out of the way this is two fp64
       implementations of one expression: summation order only, measured
       at 4.4e-16 (hidden=16) and 2.0e-14 (hidden=2048, where the mean
       has 2048 terms). 1e-12 is ~50x the mid reading — the same ratio
       the full-model fp64 stages use — and still five orders under the
       island, so an fp32 value leaking into the port cannot hide here. *)
    gate (label ^ " (fp64)") ours ref64 1e-12 1e-12
  in
  run_case "tiny";
  run_case "mid";

  (* ---------------- D1b: differential vs GPT-2 LayerNorm ---------------- *)
  (* The three differences between the ops, and the identity that makes
     them exhaustive. Same (x, w) as D1a; eps placement is D1c's half,
     which the identity below also pins — it holds only with eps inside
     the sqrt. *)
  let x, w, _, _, _ = load_case "tiny" in
  let d = Array.length w in
  let ln_of ~scale ~shift ~eps_placement =
    Norm.{ scale; shift; eps = eps_cfg; eps_placement }
  in

  (* (i) no mean subtraction. Remove the mean and the two ops must become
     the same function: var and mean(x^2) agree on a zero-mean row, and
     both sides use eps inside the sqrt with no shift. 1e-12 is ~1000x
     fp64 noise at these magnitudes and far under the 5e-6 that a wrong
     eps placement would produce. *)
  let x0 = Array.map (fun row -> Array.map (fun v -> v -. mean_vec row) row) x in
  let rms0 = Norm.rms_norm w ~eps:eps_cfg x0 in
  let ln0 = Norm.layer_norm (ln_of ~scale:w ~shift:(zeros_vec d) ~eps_placement:Norm.Eps_inside_sqrt) x0 in
  let identity_gap = max_abs_diff_mat rms0 ln0 in
  check "d1b_zero_mean_row_collapses_the_difference" (identity_gap < 1e-12);
  Printf.printf "    |rms_norm(x-mean) - layer_norm(x-mean)| = %.3e\n" identity_gap;

  (* And the difference is real: LayerNorm is shift-*invariant* by
     construction, RMSNorm is not. An RMSNorm that quietly subtracted the
     mean would be shift-invariant and fail this — which is the mistake a
     comparison against the reference does not always catch. *)
  let c = 0.5 in
  let x_shift = Array.map (fun row -> Array.map (fun v -> v +. c) row) x in
  let ln_x = ln_of ~scale:w ~shift:(zeros_vec d) ~eps_placement:Norm.Eps_inside_sqrt in
  check "d1b_layernorm_is_shift_invariant"
    (max_abs_diff_mat (Norm.layer_norm ln_x x_shift) (Norm.layer_norm ln_x x) < 1e-12);
  check_diverges "d1b_rmsnorm_is_shift_sensitive" ~min_gap:1e-3
    (Norm.rms_norm w ~eps:eps_cfg x_shift) (Norm.rms_norm w ~eps:eps_cfg x);

  (* (ii) no bias. A zero weight is not the identity for a norm with a
     bias term and is for RMSNorm: [y = 0 * anything] leaves nothing. A
     kept shift would survive here as the shift itself. *)
  check "d1b_zero_weight_zeroes_rmsnorm"
    (mat_max_abs (Norm.rms_norm (zeros_vec d) ~eps:eps_cfg x) = 0.0);
  check "d1b_zero_scale_leaves_layernorm_shift"
    (mat_max_abs (Norm.layer_norm (ln_of ~scale:(zeros_vec d) ~shift:(ones_vec d) ~eps_placement:Norm.Eps_inside_sqrt) x) = 1.0);

  (* ---------------- D1c: eps placement ---------------- *)
  (* The convention is a name, not a decorative field: flipping it must
     change the number. Measured gap is ~2e-6 on these inputs
     (sqrt(v+eps) - sqrt(v) - eps ~ eps/(2*sqrt(v)) - eps, and v ~ 1),
     against fp64 noise of ~1e-16. *)
  let rms_outside =
    Norm.norm_forward ~scale:w ~shift:None ~eps:eps_cfg ~centering:Norm.No_centering
      ~eps_placement:Norm.Eps_outside_sqrt x
  in
  check_diverges "d1c_eps_placement_flips_rmsnorm" ~min_gap:1e-9
    (Norm.rms_norm w ~eps:eps_cfg x) rms_outside;
  (* [rms_norm] must be the inside-the-sqrt half, bit for bit — the name
     is the contract with the caller. *)
  check "d1c_rmsnorm_is_the_inside_sqrt_convention"
    (max_abs_diff_mat
       (Norm.rms_norm w ~eps:eps_cfg x)
       (Norm.norm_forward ~scale:w ~shift:None ~eps:eps_cfg ~centering:Norm.No_centering
          ~eps_placement:Norm.Eps_inside_sqrt x)
    = 0.0);
  (* Same variant, reached through the LayerNorm record: the HF-loaded
     checkpoint convention and the repo's own must disagree. *)
  check_diverges "d1c_eps_placement_flips_layernorm" ~min_gap:1e-9
    (Norm.layer_norm (ln_of ~scale:w ~shift:(zeros_vec d) ~eps_placement:Norm.Eps_inside_sqrt) x)
    (Norm.layer_norm (ln_of ~scale:w ~shift:(zeros_vec d) ~eps_placement:Norm.Eps_outside_sqrt) x);

  (* ---------------- D1d: accessor wiring ---------------- *)
  (* The complementary positive test to step A's [cfg_llama_only_raises]:
     the shared accessor returns each family's own value. The two must be
     *distinguishable* or the test proves nothing — tiny-random Llama is
     1e-6, GPT-2 is 1e-5, and the file carries the Llama one. *)
  check "d1d_norm_eps_llama_is_rms_norm_eps" (eps_cfg = 1e-6);
  check "d1d_norm_eps_gpt2_is_layer_norm_eps"
    (Config.norm_eps Config.gpt_config_124m = 1e-5
    && Config.norm_eps Config.gpt_tiny_test_config = 1e-5
    && Config.norm_eps Config.llama_tiny_test_config = 1e-5);
  let _, _, eps_file, _, _ = load_case "tiny" in
  (* fp32 precision, not fp64: the file stores eps as F32, so the value
     that came back is the config's 1e-6 to within one fp32 ulp (~2.5e-8
     relative). Asking for tighter would be asking the file to hold a
     number the format cannot represent. *)
  check "d1d_norm_eps_is_what_the_op_ran_with"
    (abs_float (eps_cfg -. eps_file) <= 1e-6 *. eps_file);
  Printf.printf "    Config.norm_eps=%.9g  file eps=%.9g  gpt2=%.9g\n" eps_cfg eps_file
    (Config.norm_eps Config.gpt_config_124m);

  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)