(** SwiGLU gates D2a-D2e (ROADMAP step D2).

    Usage: [./test_mlp <mlp_ref.safetensors>]
    (written by [ocaml/parity/export_mlp.py], which instantiates
    [LlamaMLP] rather than restating its formula).

    SwiGLU is the second Llama delta and the first with internal
    structure: [down(silu(gate(x)) * up(x))] — three projections, a
    gated product, no biases. It is still not composed: the call sites
    (pre-LN -> FFN -> residual) land in step E, and [Block] is untouched.

    | D2a | vs [LlamaMLP] at hidden 16/inter 64 and the square 256/256,
           dual abs+rel |
    | D2b | differential vs the GPT-2 FFN: a zero input zeroes ours and
           not theirs (no bias), and the up path is load-bearing |
    | D2c | silu stability on the extreme grid, where the naive sigmoid
           overflows or flushes to zero |
    | D2d | [Config.ffn_inner] is [intermediate_size] for Llama, and is
           the shape D2a's weights were checked against |
    | D2e | GELU on the gate is a different function, so the activation
           choice is observable rather than a naming difference |

    D2a compares against the file's own inputs and weights, so it cannot
    pass by agreeing with PyTorch on the wrong numbers. D2b is the
    differential half: the GPT-2 FFN in [Block] is this repo's own
    two-projection, biased, GELU counterpart, and the same input through
    both must differ in exactly the three named ways. *)

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
(* Numeric helpers — the same as [test_norm.ml]'s, so that a number
   printed here means the same thing as one printed there.            *)
(* ------------------------------------------------------------------ *)

let mat_max_abs (m : Tensor.mat) : float =
  let v = ref 0.0 in
  Array.iter (Array.iter (fun x -> v := max !v (abs_float x))) m;
  !v

let max_abs_diff_mat (a : Tensor.mat) (b : Tensor.mat) : float =
  let rows = Array.length a in
  if rows <> Array.length b then failwith "test_mlp: row count mismatch";
  let m = ref 0.0 in
  for i = 0 to rows - 1 do
    if Array.length a.(i) <> Array.length b.(i) then failwith "test_mlp: col count mismatch";
    for j = 0 to Array.length a.(i) - 1 do
      m := max !m (abs_float (a.(i).(j) -. b.(i).(j)))
    done
  done;
  !m

let vec_max_abs (v : Tensor.vec) : float =
  let m = ref 0.0 in
  Array.iter (fun x -> m := max !m (abs_float x)) v;
  !m

let max_abs_diff_vec (a : Tensor.vec) (b : Tensor.vec) : float =
  if Array.length a <> Array.length b then failwith "test_mlp: vec length mismatch";
  let m = ref 0.0 in
  Array.iter2 (fun x y -> m := max !m (abs_float (x -. y))) a b;
  !m

(** The numeric stage line, in [parity.ml]'s format and with its dual
    abs+rel criterion. The dual gate is not decoration: the reference is
    fp32 — [LlamaMLP] casts to float32 before reducing — so where |y|
    reaches ~2, one fp32 ulp is already ~1e-7 absolute and the
    accumulation noise over 512 terms is ~1e-6, which a flat absolute
    gate would either measure as the reference's own rounding or fail
    on. The relative criterion pins the computation; the absolute one
    still catches a whole-tensor shift that relative error hides at
    small magnitudes. *)
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
    [min_gap] is the point of the test: fp64 noise on a value of order 1
    is ~1e-16, so a threshold of 1e-3 is "structurally different" while
    still leaving room for the difference being genuinely small. *)
let check_diverges name ~min_gap a b =
  let d = max_abs_diff_mat a b in
  let ok = d > min_gap in
  check (Printf.sprintf "%s (max_abs_diff=%.3e, must exceed %.0e)" name d min_gap) ok

let () =
  if Array.length Sys.argv <> 2 then (
    Printf.eprintf "usage: %s <mlp_ref.safetensors>\n" Sys.argv.(0);
    exit 2);
  let f = Safetensors.open_file Sys.argv.(1) in

  let load_case name =
    let x = Safetensors.mat_of f (Printf.sprintf "d2.%s.x" name) in
    let w_gate = Safetensors.mat_of f (Printf.sprintf "d2.%s.gate_proj" name) in
    let w_up = Safetensors.mat_of f (Printf.sprintf "d2.%s.up_proj" name) in
    let w_down = Safetensors.mat_of f (Printf.sprintf "d2.%s.down_proj" name) in
    let ref = Safetensors.mat_of f (Printf.sprintf "d2.%s.ref" name) in
    if Array.length x = 0 || Array.length w_gate = 0 then failwith "test_mlp: empty case";
    if Array.length w_gate.(0) <> Array.length x.(0) then
      failwith "test_mlp: gate_proj's d_in does not match x's last dim";
    (x, w_gate, w_up, w_down, ref)
  in

  (* ---------------- D2a: vs LlamaMLP ---------------- *)
  Printf.printf "D2a: swiglu vs LlamaMLP\n";
  let run_case label =
    let x, w_gate, w_up, w_down, ref = load_case label in
    let ours = Mlp.forward Mlp.{ gate_proj = w_gate; up_proj = w_up; down_proj = w_down; } x in
    gate label ours ref 1e-5 1e-5;
    Printf.printf "    hidden=%d intermediate=%d |ref|max=%.3e\n" (Array.length x.(0))
      (Array.length w_gate) (mat_max_abs ref)
  in
  run_case "tiny";
  run_case "mid";

  let x, w_gate, w_up, w_down, _ = load_case "tiny" in
  let hidden = Array.length x.(0) in
  let inter = Array.length w_gate in
  let mlp = Mlp.{ gate_proj = w_gate; up_proj = w_up; down_proj = w_down; } in

  (* ---------------- D2b: differential vs the GPT-2 FFN ---------------- *)
  (* The three structural differences between the two FFNs, each made
     observable on the same input. (ii) SwiGLU vs GELU is D2e's half —
     a differential against a *different architecture* cannot see an
     activation swap inside our own op, which is exactly why D2e
     exists. *)

  (* (iii) no bias. With no bias anywhere, a zero input has nowhere to
     come from: the output is exactly zero. The GPT-2 FFN on the same
     zero input is NOT zero — its two biases survive — which is the
     differential that makes our zero structural rather than the
     signature of an op that always returns zero. *)
  let zeros = Array.make_matrix (Array.length x) hidden 0.0 in
  check "d2b_zero_input_zeroes_swiglu" (mat_max_abs (Mlp.forward mlp zeros) = 0.0);
  (* The GPT-2 side with *nonzero* biases — [Attention.linear_make]
     inits bias to zeros, which would make this differential vacuous:
     a zero bias is the one case where the two FFNs agree on a zero
     input. The biases are the difference being demonstrated, so they
     are built explicitly. *)
  let rs = Random.State.make [| 0 |] in
  let lin d_in d_out =
    let w = randn_mat rs d_out d_in in
    let b = (randn_mat rs 1 d_out).(0) in
    { Attention.w; b = Some b; d_in; d_out }
  in
  let gpt2_ffn : Block.ffn =
    { l1 = lin hidden inter; l2 = lin inter hidden; activation = Norm.Gelu_exact }
  in
  check_diverges "d2b_gpt2_ffn_zero_input_is_not_zero" ~min_gap:1e-3
    (Block.ffn_forward gpt2_ffn zeros) zeros;

  (* (i) three projections vs two: the up path is load-bearing. Zero it
     and the gated product is zero; drop the multiply from the kernel
     and the two disagree. *)
  let mlp_no_up = Mlp.{ mlp with up_proj = Array.make_matrix inter hidden 0.0; } in
  check "d2b_zero_up_zeroes_the_product" (mat_max_abs (Mlp.forward mlp_no_up x) = 0.0);
  check_diverges "d2b_up_path_is_load_bearing" ~min_gap:1e-3 (Mlp.forward mlp x)
    (Mlp.forward mlp_no_up x);

  (* Same input through both FFNs: they must differ by more than fp
     noise — they are different functions, not the same one with
     different weights. *)
  check_diverges "d2b_differs_from_gpt2_ffn" ~min_gap:1e-3 (Mlp.forward mlp x)
    (Block.ffn_forward gpt2_ffn x);

  (* ---------------- D2c: silu stability ---------------- *)
  (* The naive sigmoid [1 / (1 + exp(-x))] overflows fp64 for
     [x < -709] — where the true sigmoid is still [~1e-308] — and its
     reciprocal form is inf/inf = nan for [x > 709]. The grid is where
     those failures live. -710 is deliberately NOT compared against
     torch: torch's fp32 SiLU flushes it to -0.0 while the true value
     is [~-7e-306], so the fp64 kernel is pinned to the *function*
     there, stricter than the fp32 reference — the same reasoning as
     [Norm.rms_norm] accumulating in fp64. *)
  let silu_x = Safetensors.vec_of f "d2.silu.x" in
  let silu_ref = Safetensors.vec_of f "d2.silu.ref" in
  let silu_ours = Mlp.silu_vec silu_x in
  let silu_finite = Array.for_all (fun v -> not (Float.is_nan v) && not (Float.is_infinite v)) in
  check "d2c_silu_grid_is_finite" (silu_finite silu_ours);
  (* vs torch, on the points where the fp32 reference can represent
     the answer: abs 1e-6 covers the fp32 rounding of values up to ~20
     (one ulp at 20 is ~2e-6... the grid's largest is 800, where both
     sides are exact). *)
  check "d2c_silu_matches_torch" (max_abs_diff_vec silu_ours silu_ref <= 1e-6);
  Printf.printf "    max|ours-ref| on the grid = %.3e\n" (max_abs_diff_vec silu_ours silu_ref);
  (* The flushed points, where the naive form says 0 or -0 and the
     function says a tiny negative: ours must be strictly negative. A
     fp32-rounded naive sigmoid returns +/-0.0 here — [0.0 < 0.0] and
     [-0.0 < 0.0] are both false, so the sign check is the catcher. *)
  check "d2c_silu_neg100_is_strictly_negative" (Mlp.silu_scalar (-100.0) < 0.0);
  check "d2c_silu_neg710_is_strictly_negative" (Mlp.silu_scalar (-710.0) < 0.0);
  Printf.printf "    silu(-100)=%.3e  silu(-710)=%.3e  silu(100)=%.1f  silu(800)=%.1f\n"
    (Mlp.silu_scalar (-100.0)) (Mlp.silu_scalar (-710.0)) (Mlp.silu_scalar 100.0)
    (Mlp.silu_scalar 800.0);
  check "d2c_silu_pos100_is_100" (Mlp.silu_scalar 100.0 = 100.0);
  check "d2c_silu_zero_is_zero" (Mlp.silu_scalar 0.0 = 0.0);

  (* ---------------- D2d: accessor wiring ---------------- *)
  (* The complementary positive test to D1d's: the shared accessor
     returns each family's own width. The two must be *distinguishable*
     or the test proves nothing — [llama_tiny_test_config] is the case
     where they are (intermediate 128 vs 4x64 = 256). *)
  let cfg = Config.llama_tiny_test_config in
  check "d2d_ffn_inner_llama_is_intermediate_size"
    (Config.ffn_inner cfg = 128 && Config.ffn_inner cfg <> 4 * cfg.Config.hidden_size);
  check "d2d_ffn_inner_llama_tiny_random_is_intermediate_size"
    (Config.ffn_inner Config.llama_tiny_random_config = 64);
  check "d2d_ffn_inner_gpt2_is_4x_hidden"
    (Config.ffn_inner Config.gpt_config_124m = 4 * 768
     && Config.ffn_inner Config.gpt_tiny_test_config = 4 * 16);
  (* And the file's weights are the shape the accessor claims — the
     wiring D2a actually ran with. The tiny case is tiny-random-Llama's
     own shape (hidden 16, intermediate 64). *)
  let cfg_tr = Config.llama_tiny_random_config in
  check "d2d_d2a_weights_match_ffn_inner"
    (Array.length w_gate = Config.ffn_inner cfg_tr
     && Array.length w_gate.(0) = cfg_tr.Config.hidden_size
     && Array.length w_down = cfg_tr.Config.hidden_size
     && Array.length w_down.(0) = Config.ffn_inner cfg_tr);
  Printf.printf "    Config.ffn_inner: tiny_test=%d tiny_random=%d gpt2_124m=%d\n"
    (Config.ffn_inner cfg) (Config.ffn_inner cfg_tr) (Config.ffn_inner Config.gpt_config_124m);

  (* ---------------- D2e: the activation choice is observable ---------- *)
  (* silu(gate) * up and gelu(gate) * up are different functions on the
     same input — so "GELU on the gate" is a mutation these gates can
     see, not a naming difference. This is the mutation D2b cannot
     catch: a differential against a different architecture says
     nothing about which activation our own op uses. *)
  let g = linear_fwd w_gate x None in
  let u = linear_fwd w_up x None in
  let h_silu = hadamard (Mlp.silu_mat g) u in
  let h_gelu = hadamard (Norm.gelu_mat g) u in
  check_diverges "d2e_silu_gate_differs_from_gelu_gate" ~min_gap:1e-3 h_silu h_gelu;
  check_diverges "d2e_gelu_on_the_gate_changes_the_output" ~min_gap:1e-3 (Mlp.forward mlp x)
    (linear_fwd w_down h_gelu None);

  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)
