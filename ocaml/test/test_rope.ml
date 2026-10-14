(** RoPE gates D3a-D3d (ROADMAP step D3).

     Usage: [./test_rope <rope_ref.safetensors>]
     (written by [ocaml/parity/export_rope.py], which uses the repo's own
     [RoPE] class rather than restating its formula).

     RoPE is the third Llama delta. The gates prove it in isolation before
     it is composed with attention in step E:

     | D3a | vs [RoPE.apply] on random Q/K — numerical parity |
     | D3b | relative-position invariance: <R_i q, R_j k> == <R_{i+d} q, R_{j+d} k> |
     | D3c | direct 2D rotation matrix == apply (cos*x + sin*rotate_half) |
     | D3d | cached-at-pos-p (position_ids=[p]) == full-forward row p |

     D3a compares against the file's own inputs, so it cannot pass by
     agreeing with PyTorch on the wrong numbers. D3b is the algebraic
     property that makes RoPE a relative-position scheme — it catches any
     implementation that rotates by the wrong angle or the wrong pairing,
     even if the reference comparison somehow agreed. D3c is the explicit
     rotation-matrix form; if `apply` and `apply_direct` disagree, one of
     them has a bug in rotate_half or the cos/sin indexing. D3d is the
     KV-cache decode path: it must reproduce the prefill result exactly,
     or generation will diverge from training-time positions. *)

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

(* ---------- numeric helpers (same as test_norm.ml / parity.ml) ---------- *)

let mat_max_abs (m : Tensor.mat) : float =
  let v = ref 0.0 in
  Array.iter (Array.iter (fun x -> v := max !v (abs_float x))) m;
  !v

let max_abs_diff_mat (a : Tensor.mat) (b : Tensor.mat) : float =
  let rows = Array.length a in
  if rows <> Array.length b then failwith "test_rope: row count mismatch";
  let m = ref 0.0 in
  for i = 0 to rows - 1 do
    if Array.length a.(i) <> Array.length b.(i) then failwith "test_rope: col count mismatch";
    for j = 0 to Array.length a.(i) - 1 do
      m := max !m (abs_float (a.(i).(j) -. b.(i).(j)))
    done
  done;
  !m

let gate name ours refm abs_tol rel_tol =
  let diff = max_abs_diff_mat ours refm in
  let refmag = mat_max_abs refm in
  let rel = diff /. max refmag 1e-30 in
  let ok = diff <= abs_tol && rel <= rel_tol in
  if ok then incr passed else incr failed;
  Printf.printf "%-14s max_abs_diff=%.3e ref_max=%.3e rel=%.3e abs_tol=%.0e rel_tol=%.0e %s\n" name
    diff refmag rel abs_tol rel_tol
    (if ok then "OK" else "BREACH")

let dot (a : Tensor.vec) (b : Tensor.vec) : float =
  let s = ref 0.0 in
  for i = 0 to Array.length a - 1 do
    s := !s +. a.(i) *. b.(i)
  done;
  !s

let max_abs_diff_vec (a : Tensor.vec) (b : Tensor.vec) : float =
  let m = ref 0.0 in
  for i = 0 to Array.length a - 1 do
    m := max !m (abs_float (a.(i) -. b.(i)))
  done;
  !m

(* ---------- explicit 2D rotation (D3c independent implementation) ---------- *)

let apply_direct ~base ~head_dim ~(positions : int array) (x : mat) : mat =
  let half = head_dim / 2 in
  Array.mapi
    (fun i row ->
      let m = positions.(i) in
      Array.init head_dim (fun j ->
          let j' = if j < half then j else j - half in
          let theta = 1.0 /. (base ** (2.0 *. float_of_int j' /. float_of_int head_dim)) in
          let angle = float_of_int m *. theta in
          let c = cos angle in
          let s = sin angle in
          if j < half then
            c *. row.(j) -. s *. row.(j + half)
          else
            s *. row.(j - half) +. c *. row.(j)))
    x

let () =
  if Array.length Sys.argv < 2 then (
    Printf.eprintf "usage: %s <rope_ref.safetensors> [rope_llama3_ref.safetensors]\n" Sys.argv.(0);
    exit 2);
  let f = Safetensors.open_file Sys.argv.(1) in

  let cos = Safetensors.mat_of f "d3.cos" in
  let sin = Safetensors.mat_of f "d3.sin" in
  let q = Safetensors.mat_of f "d3.q" in
  let k = Safetensors.mat_of f "d3.k" in
  let q_ref = Safetensors.mat_of f "d3.q_ref" in
  let k_ref = Safetensors.mat_of f "d3.k_ref" in
  let q_cached = Safetensors.mat_of f "d3.q_cached" in
  let k_cached = Safetensors.mat_of f "d3.k_cached" in
  let q_cached_ref = Safetensors.mat_of f "d3.q_cached_ref" in
  let k_cached_ref = Safetensors.mat_of f "d3.k_cached_ref" in
  let pos = Int64.to_int (Safetensors.i64_array_of f "d3.pos").(0) in

  let head_dim = Array.length cos.(0) in
  let ctx_len = Array.length cos in
  let seq_len = Array.length q in
  let base = 10000.0 in

  (* Sanity: the cos/sin tables must cover every position we apply. *)
  if ctx_len <= pos then
    failwith (Printf.sprintf "test_rope: pos=%d exceeds ctx_len=%d" pos ctx_len);

  (* ---------- D3a: vs RoPE.apply (full-forward, positions 0..seq-1) ---------- *)
  Printf.printf "D3a: Rope.apply vs RoPE.apply (full-forward, positions 0..%d)\n" seq_len;
  let q_ours = Rope.apply ~cos ~sin q in
  let k_ours = Rope.apply ~cos ~sin k in
  gate "d3a_q" q_ours q_ref 1e-5 1e-5;
  gate "d3a_k" k_ours k_ref 1e-5 1e-5;

  (* ---------- D3b: relative-position invariance ---------- *)
  (* <R_i q, R_j k> == <R_{i+d} q, R_{j+d} k> for any d.
     This is the defining property of RoPE: the inner product of a rotated
     query and a rotated key depends only on their relative position.
     CRITICAL: q and k must be the SAME vectors at all positions — using
     different rows (q[i] vs q[i+d]) tests nothing, since those are
     unrelated random vectors. *)
  Printf.printf "D3b: relative-position invariance\n";
  let rs = Random.State.make [| 42; 42; 42; 42 |] in
  let q_vec = Array.init head_dim (fun _ -> Random.State.float rs 2.0 -. 1.0) in
  let k_vec = Array.init head_dim (fun _ -> Random.State.float rs 2.0 -. 1.0) in
  (* Use OCaml-computed fp64 tables for the algebraic property test —
     the file's fp32 tables introduce ~1e-7 noise that has nothing to
     with the math being tested. *)
  let cos64, sin64 = Rope.compute_angles ~base ~head_dim ~ctx_len in
  let max_inv_err = ref 0.0 in
  for i = 0 to seq_len - 1 do
    for j = 0 to seq_len - 1 do
      for d = 1 to seq_len - max i j - 1 do
        let qi = Rope.apply ~positions:(Some [| i |]) ~cos:cos64 ~sin:sin64 [| q_vec |] in
        let kj = Rope.apply ~positions:(Some [| j |]) ~cos:cos64 ~sin:sin64 [| k_vec |] in
        let lhs = dot qi.(0) kj.(0) in
        let qid = Rope.apply ~positions:(Some [| i + d |]) ~cos:cos64 ~sin:sin64 [| q_vec |] in
        let kjd = Rope.apply ~positions:(Some [| j + d |]) ~cos:cos64 ~sin:sin64 [| k_vec |] in
        let rhs = dot qid.(0) kjd.(0) in
        max_inv_err := max !max_inv_err (abs_float (lhs -. rhs))
      done
    done
  done;
  check (Printf.sprintf "d3b_relative_position_invariance (max_err=%.3e)" !max_inv_err)
    (!max_inv_err < 1e-10);

  (* ---------- D3c: direct 2D rotation == apply ---------- *)
  (* Use the OCaml-computed table (fp64) for both paths so the comparison
     tests the math, not fp32-vs-fp64 table precision. *)
  Printf.printf "D3c: apply_direct vs Rope.apply\n";
  let cos64, sin64 = Rope.compute_angles ~base ~head_dim ~ctx_len in
  let positions = Array.init seq_len (fun i -> i) in
  let q_direct = apply_direct ~base ~head_dim ~positions q in
  let q_apply = Rope.apply ~cos:cos64 ~sin:sin64 q in
  check "d3c_direct_equals_apply_q" (max_abs_diff_mat q_direct q_apply < 1e-12);
  let k_direct = apply_direct ~base ~head_dim ~positions k in
  let k_apply = Rope.apply ~cos:cos64 ~sin:sin64 k in
  check "d3c_direct_equals_apply_k" (max_abs_diff_mat k_direct k_apply < 1e-12);

  (* ---------- D3d: cached-at-pos-p == full-forward row p ---------- *)
  Printf.printf "D3d: cached-at-pos-%d == full-forward row %d\n" pos pos;
  let q_cached_ours = Rope.apply ~positions:(Some [| pos |]) ~cos ~sin q_cached in
  let k_cached_ours = Rope.apply ~positions:(Some [| pos |]) ~cos ~sin k_cached in
  gate "d3d_q_cached" q_cached_ours q_cached_ref 1e-5 1e-5;
  gate "d3d_k_cached" k_cached_ours k_cached_ref 1e-5 1e-5;

  (* Build a sequence long enough to include pos for full-forward comparison *)
  let long_seq = Array.init (pos + 1) (fun i ->
    if i = pos then Array.copy q_cached.(0)
    else Array.init head_dim (fun _ -> 0.0)) in
  let long_ours = Rope.apply ~cos ~sin long_seq in
  check "d3d_cached_equals_full_forward_row_q"
    (max_abs_diff_mat q_cached_ours (Array.sub long_ours pos 1) < 1e-12);

  let long_kseq = Array.init (pos + 1) (fun i ->
    if i = pos then Array.copy k_cached.(0)
    else Array.init head_dim (fun _ -> 0.0)) in
  let long_kours = Rope.apply ~cos ~sin long_kseq in
  check "d3d_cached_equals_full_forward_row_k"
    (max_abs_diff_mat k_cached_ours (Array.sub long_kours pos 1) < 1e-12);

  (* ---------- D4: llama3 scaling ---------- *)
  if Array.length Sys.argv >= 3 then (
    let f4 = Safetensors.open_file Sys.argv.(2) in
    let inv_freq_ref = Safetensors.vec_of f4 "d4.inv_freq" in
    let base_inv_freq_ref = Safetensors.vec_of f4 "d4.base_inv_freq" in
    let cos_ref = Safetensors.mat_of f4 "d4.cos" in
    let sin_ref = Safetensors.mat_of f4 "d4.sin" in

    let head_dim4 = Array.length cos_ref.(0) in
    let ctx_len4 = Array.length cos_ref in
    let scaling = Config.Llama3 {
      factor = 8.0;
      low_freq_factor = 1.0;
      high_freq_factor = 4.0;
      original_max_position_embeddings = 8192;
    } in

    Printf.printf "D4: Rope.compute_angles_llama3 vs _compute_llama3_parameters\n";
    let cos4, sin4 =
      Rope.compute_angles_llama3 ~base ~head_dim:head_dim4 ~ctx_len:ctx_len4 ~scaling
    in
    gate "d4_cos" cos4 cos_ref 1e-5 1e-5;
    gate "d4_sin" sin4 sin_ref 1e-5 1e-5;

    (* Verify the scaling actually changed frequencies (not identity) *)
    let max_inv_diff = max_abs_diff_vec inv_freq_ref base_inv_freq_ref in
    check (Printf.sprintf "d4_scaling_changes_frequencies (max_diff=%.3e)" max_inv_diff)
      (max_inv_diff > 1e-4);

    (* Verify llama3-scaled tables differ from base RoPE tables *)
    let cos_base, _ =
      Rope.compute_angles ~base ~head_dim:head_dim4 ~ctx_len:ctx_len4
    in
    let cos_diff = max_abs_diff_mat cos4 cos_base in
    check (Printf.sprintf "d4_llama3_differs_from_base (max_diff=%.3e)" cos_diff)
      (cos_diff > 1e-4)
  );

  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)
