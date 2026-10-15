(** GQA gates D5a-D5f (ROADMAP step D5).

     Usage: [./test_gqa <gqa_ref.safetensors>]
     (written by [ocaml/parity/export_gqa.py], which instantiates HF's
     [LlamaAttention] and the repo's own [GroupedQueryAttention]).

     Grouped-Query Attention is the last Llama delta before composition:
     K and V are projected onto [num_kv_groups] heads instead of
     [num_heads], and each K/V head serves [num_repeat] query heads.

     | D5a | vs HF [LlamaAttention] on the same weights/inputs |
     | D5b | vs the repo's own [GroupedQueryAttention] — the actual port contract |
     | D5c | head grouping is **contiguous** ([repeat_interleave], query head [h]
     |       reads kv head [h / num_repeat]), NOT tiled ([h mod num_kv_groups]) |
     | D5d | KV cache holds [num_kv_groups] heads (not [num_heads]), refuses a
     |       wider push, and cached decode at position p ≡ full-forward row p |
     | D5e | [num_kv_groups = num_heads] makes the expansion the identity |
     | D5f | out_proj's bias is load-bearing; Q/K/V carry none |

     D5c is the gate that earns its keep. Grouping is a *permutation*:
     tiling ([h mod kv]) and interleaving ([h / repeat]) both hand every
     query head some kv head, both keep the shapes right, and both
     produce plausible numbers. Nothing raises. The only thing separating
     them is that the two assignments are genuinely different for
     kv_groups > 1 — so D5c asserts our output *differs* from the tiled
     one, and that the reference is the closer of the two. Without that,
     swapping the convention would surface as an unexplained numerical
     breach rather than as "the grouping was wrong".

     The shapes here are [Config.llama_tiny_test_config]'s: 4 heads /
     2 kv heads. tiny-random-LlamaForCausalLM has 4/4 and therefore no
     GQA at all, so a green step E is not a green D5 (see ROADMAP.md). *)

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
  if rows <> Array.length b then failwith "test_gqa: row count mismatch";
  let m = ref 0.0 in
  for i = 0 to rows - 1 do
    if Array.length a.(i) <> Array.length b.(i) then failwith "test_gqa: col count mismatch";
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

let check_diverges name ~min_gap a b =
  let d = max_abs_diff_mat a b in
  let ok = d > min_gap in
  check (Printf.sprintf "%s (max_abs_diff=%.3e, must exceed %.0e)" name d min_gap) ok

(** Build a [linear] from a reference weight (+ optional bias). *)
let mk_linear (w : mat) (b : vec option) : Attention.linear =
  let d_out = Array.length w in
  let d_in = if d_out = 0 then 0 else Array.length w.(0) in
  { Attention.w; b; d_in; d_out }

(** The *wrong* head grouping, written out here so D5c can assert our
    output does not equal it. [h mod kv] tiles kv heads across query
    heads instead of giving each kv head a contiguous block. *)
let repeat_kv_heads_tiled (a : Attention.gqa) (hs : mat array) : mat array =
  Array.init a.Attention.num_heads (fun h -> hs.(h mod a.Attention.num_kv_groups))

(** Per-head attention over *already-expanded* K/V, then merge + project.
    Written once and shared by the tiled path (D5c) and the degenerate
    identity path (D5e), so neither re-implements the combine stage. *)
let combine_expanded (a : Attention.gqa) (qh : mat array) (k_full : mat array) (v_full : mat array) :
    mat =
  let ctx =
    Array.init a.Attention.num_heads (fun h ->
        Attention.head_forward ~q:qh.(h) ~k_all:k_full.(h) ~v_all:v_full.(h) ~attn_mask:None)
  in
  Attention.linear_forward_batch a.Attention.out_proj (Attention.merge_heads ctx)

let () =
  if Array.length Sys.argv <> 2 then (
    Printf.eprintf "usage: %s <gqa_ref.safetensors>\n" Sys.argv.(0);
    exit 2);
  let f = Safetensors.open_file Sys.argv.(1) in

  let x = Safetensors.mat_of f "d5.x" in
  let w_q = Safetensors.mat_of f "d5.w_q" in
  let w_k = Safetensors.mat_of f "d5.w_k" in
  let w_v = Safetensors.mat_of f "d5.w_v" in
  let w_o = Safetensors.mat_of f "d5.w_o" in
  let b_o = Safetensors.vec_of f "d5.b_o" in
  let ref_hf = Safetensors.mat_of f "d5.ref_hf" in
  let ref_repo = Safetensors.mat_of f "d5.ref_repo" in

  let num_heads = 4 in
  let num_kv_groups = 2 in
  (* Weights are [(out, in)] project-wide, so rows are the output dim.
     w_q is [num_heads * head_dim, d_in]; w_k/w_v are
     [num_kv_groups * head_dim, d_in] — the narrower row count is exactly
     what makes GQA (and the KV cache) kv_heads-sized. *)
  let d_in = if Array.length w_q = 0 then 0 else Array.length w_q.(0) in
  let d_out = Array.length w_q in
  let head_dim = d_out / num_heads in

  (* The file's K/V projection width must be kv_groups * head_dim — that is
     what makes the cache kv-sized, and num_heads * head_dim would mean the
     reference is not GQA at all. *)
  let expect_kv_width = num_kv_groups * head_dim in
  if Array.length w_k <> expect_kv_width || Array.length w_v <> expect_kv_width then
    failwith
      (Printf.sprintf "test_gqa: w_k/w_v rows are %d/%d, expected num_kv_groups * head_dim = %d"
         (Array.length w_k) (Array.length w_v) expect_kv_width);
  if Array.length w_k.(0) <> d_in || Array.length ref_hf.(0) <> d_out then
    failwith "test_gqa: reference shapes disagree with the declared GQA config";

  let base = 10000.0 in
  let ctx_len = Array.length x in

  let a : Attention.gqa =
    {
      Attention.w_q = mk_linear w_q None;
      w_k = mk_linear w_k None;
      w_v = mk_linear w_v None;
      out_proj = mk_linear w_o (Some b_o);
      num_heads;
      num_kv_groups;
      head_dim;
      d_out;
    }
  in

  let cos, sin = Rope.compute_angles ~base ~head_dim ~ctx_len in

  (* ---------------- D5a / D5b: parity ---------------- *)
  Printf.printf "D5a: gqa_forward vs LlamaAttention / GroupedQueryAttention\n";
  let ours = Attention.gqa_forward ~cos ~sin a x in
  gate "d5a_vs_hf" ours ref_hf 1e-5 1e-5;
  gate "d5b_vs_repo" ours ref_repo 1e-5 1e-5;

  (* ---------------- D5c: grouping is contiguous, not tiled ---------------- *)
  Printf.printf "D5c: head grouping convention\n";
  let qh, kh, vh = Attention.gqa_project ~cos ~sin a x in
  let tiled = combine_expanded a qh (repeat_kv_heads_tiled a kh) (repeat_kv_heads_tiled a vh) in
  let d_tiled = max_abs_diff_mat ours tiled in
  let d_ref = max_abs_diff_mat ours ref_hf in
  Printf.printf "    diff vs tiled=%.3e   diff vs reference=%.3e\n" d_tiled d_ref;
  check_diverges "d5c_tiling_must_not_agree" ~min_gap:1e-3 ours tiled;
  check "d5c_reference_is_closer_than_tiling" (d_ref <= d_tiled);
  check "d5c_num_repeat" (Attention.num_repeat a = num_heads / num_kv_groups);

  (* ---------------- D5d: KV cache is kv_heads-sized ---------------- *)
  Printf.printf "D5d: KV cache sized by kv_heads, not num_heads\n";
  check "d5d_project_yields_kv_groups_heads"
    (Array.length kh = num_kv_groups && Array.length vh = num_kv_groups);
  let wide = Array.make num_heads (Array.copy kh.(0)) in
  (* [Kv_cache.push] only compares head counts against an *existing* entry,
     so a num_heads-wide push into a fresh cache is accepted by design —
     the cache is shape-agnostic. The check that matters is pushing it on
     top of a real kv_groups entry, which is exactly the accident D5 is
     about (expanding to num_heads before pushing). *)
  let raised_push =
    try
      let c = Kv_cache.create ~num_layers:1 in
      Kv_cache.push c ~layer_idx:0 ~keys:kh ~values:vh;
      Kv_cache.push c ~layer_idx:0 ~keys:wide ~values:wide;
      false
    with Invalid_argument _ -> true
  in
  check "d5d_cache_rejects_num_heads_wide_push" raised_push;
  let raised_fwd =
    try
      ignore
        (Attention.gqa_forward ~kv:(Some { Kv_cache.keys = wide; values = wide }) ~cos ~sin a x);
      false
    with Invalid_argument _ -> true
  in
  check "d5d_forward_rejects_num_heads_wide_cache" raised_fwd;

  (* Prefill then decode: cached step at position p must equal the
     full-forward row p. *)
  let prefill_len = ctx_len - 1 in
  let prefill_x = Array.sub x 0 prefill_len in
  let _, kh_pre, vh_pre = Attention.gqa_project ~cos ~sin a prefill_x in
  let cache = Kv_cache.create ~num_layers:1 in
  Kv_cache.push cache ~layer_idx:0 ~keys:kh_pre ~values:vh_pre;
  check "d5d_past_len_after_prefill" (Kv_cache.past_len cache 0 = prefill_len);
  let peeked = match Kv_cache.peek cache 0 with Some kv -> kv | None -> assert false in
  check "d5d_cache_head_count_is_kv_groups"
    (Array.length peeked.Kv_cache.keys = num_kv_groups);
  let last_row = [| Array.copy x.(ctx_len - 1) |] in
  let decoded =
    Attention.gqa_forward ~kv:(Some peeked) ~positions:(Some [| prefill_len |]) ~cos ~sin a last_row
  in
  (* The full-forward reference must put that same row at position
     prefill_len — which is not row 0, so compare against a prefill that
     actually reaches that position. *)
  let full_x = Array.sub x 0 (prefill_len + 1) in
  let full = Attention.gqa_forward ~cos ~sin a full_x in
  let full_last = [| full.(prefill_len) |] in
  check "d5d_cached_decode_equals_full_forward" (max_abs_diff_mat decoded full_last < 1e-12);

  (* ---------------- D5e: kv_groups = num_heads degenerates ---------------- *)
  Printf.printf "D5e: GQA with kv_groups = num_heads\n";
  let degenerate : Attention.gqa =
    { a with Attention.w_k = mk_linear w_q None; w_v = mk_linear w_q None; num_kv_groups = num_heads }
  in
  check "d5e_num_repeat_is_one" (Attention.num_repeat degenerate = 1);
  (* num_repeat = 1 makes the expansion the identity, so the combine
     stage must equal the un-expanded per-head attention exactly. *)
  let dq, dkh, dvh = Attention.gqa_project ~cos ~sin degenerate x in
  let unexpanded = combine_expanded degenerate dq dkh dvh in
  let expanded = Attention.gqa_forward ~cos ~sin degenerate x in
  check "d5e_identity_expansion" (max_abs_diff_mat unexpanded expanded < 1e-12);
  check "d5e_repeat_kv_heads_is_identity" (Attention.repeat_kv_heads degenerate dkh == dkh);

  (* ---------------- D5f: bias structure ---------------- *)
  Printf.printf "D5f: bias structure\n";
  (* Asserted on the production constructor, not on a record this test
     builds itself: [gqa_make] is what ships, and its bias structure is
     a real claim (q/k/v None, out_proj Some) that a flipped flag would
     silently break. *)
  let built = Attention.gqa_make (Random.State.make [| 7 |]) ~d_in ~d_out ~num_heads ~num_kv_groups () in
  check "d5f_qkv_carry_no_bias"
    (built.Attention.w_q.Attention.b = None && built.Attention.w_k.Attention.b = None
     && built.Attention.w_v.Attention.b = None);
  check "d5f_out_proj_carries_bias" (built.Attention.out_proj.Attention.b <> None);
  let zero_bias = Array.make d_out 0.0 in
  let with_zero = Attention.gqa_forward ~cos ~sin { a with Attention.out_proj = mk_linear w_o (Some zero_bias) } x in
  let with_none = Attention.gqa_forward ~cos ~sin { a with Attention.out_proj = mk_linear w_o None } x in
  check_diverges "d5f_out_proj_bias_is_load_bearing" ~min_gap:1e-6 ours with_zero;
  check "d5f_zero_bias_equals_no_bias" (max_abs_diff_mat with_zero with_none < 1e-12);

  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)