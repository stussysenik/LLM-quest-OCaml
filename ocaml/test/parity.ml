(** Three-layer parity harness: OCaml GPT-2 vs PyTorch reference.

    Usage: [./parity_check <gpt2_with_ref.safetensors>]
    (file written by [ocaml/parity/export_reference.py]).

    Gates (from [ocaml/ROADMAP.md]):
    - embeddings (hidden.0) match to ~1e-6  → weight loader correct
    - block outputs (hidden.1..11) to ~1e-5 → architecture correct
    - final logits to ~1e-4                 → fp32 accumulation over 12 layers

    Two calibration lessons from the first run against real weights:
    - GPT-2 has outlier dims (~3000); fp32-vs-fp64 summation noise on
      those is ~4e-4 absolute (~1e-7 relative), so block gates use a
      dual abs+rel criterion rather than a flat 1e-5.
    - transformers 5.x appends the POST-ln_f state as hidden_states[-1]
      (verified: hidden.12 @ lm_head.T == logits, and its std is ~5,
      not ~80 like the raw outlier-dominated states). So hidden.12 is
      compared against our final-LN output, not the raw block output.

    The first stage that diverges pinpoints the bug's component. *)

let mat_max_abs (m : Tensor.mat) : float =
  let v = ref 0.0 in
  Array.iter (Array.iter (fun x -> v := max !v (abs_float x))) m;
  !v

(** Top-3 absolute divergences with locations and both values. *)
let top_diffs (a : Tensor.mat) (b : Tensor.mat) : (float * int * int * float * float) list =
  let acc = ref [] in
  for i = 0 to Array.length a - 1 do
    for j = 0 to Array.length a.(i) - 1 do
      let d = abs_float (a.(i).(j) -. b.(i).(j)) in
      acc := (d, i, j, a.(i).(j), b.(i).(j)) :: !acc
    done
  done;
  let sorted = List.sort (fun (d1, _, _, _, _) (d2, _, _, _, _) -> compare d2 d1) !acc in
  let rec take n l = if n = 0 then [] else match l with [] -> [] | x :: xs -> x :: take (n - 1) xs in
  take 3 sorted

let max_abs_diff_mat (a : Tensor.mat) (b : Tensor.mat) : float =
  let rows = Array.length a in
  if rows <> Array.length b then failwith "Parity: row count mismatch";
  let m = ref 0.0 in
  for i = 0 to rows - 1 do
    if Array.length a.(i) <> Array.length b.(i) then failwith "Parity: col count mismatch";
    for j = 0 to Array.length a.(i) - 1 do
      m := max !m (abs_float (a.(i).(j) -. b.(i).(j)))
    done
  done;
  !m

let () =
  if Array.length Sys.argv <> 2 then (
    Printf.eprintf "usage: %s <gpt2_with_ref.safetensors>\n" Sys.argv.(0);
    exit 2);
  let path = Sys.argv.(1) in
  let model = Hf_loader.load_gpt2 path in
  let n_layers = List.length model.Gpt.blocks in
  let reference = Hf_loader.load_reference path ~n_layers in
  Printf.printf "prompt ids: [%s]\n"
    (String.concat ";" (List.map string_of_int (Array.to_list reference.Hf_loader.ids)));

  let states = Gpt.forward_states model reference.Hf_loader.ids in
  let logits = Gpt.forward model reference.Hf_loader.ids in

  let failures = ref 0 in
  (* Dual gate: absolute tolerance for quiet dims, relative-to-magnitude
     tolerance for the outlier dims (~3000, fp32 accumulation noise). *)
  let gate name ours refm abs_tol rel_tol =
    let diff = max_abs_diff_mat ours refm in
    let refmag = mat_max_abs refm in
    let rel = diff /. max refmag 1e-30 in
    let ok = diff <= abs_tol && rel <= rel_tol in
    if not ok then incr failures;
    Printf.printf "%-14s max_abs_diff=%.3e ref_max=%.3e rel=%.3e abs_tol=%.0e rel_tol=%.0e %s\n" name
      diff refmag rel abs_tol rel_tol
      (if ok then "OK" else "BREACH");
    if not ok then
      List.iter
        (fun (d, i, j, o, r) ->
          Printf.printf "    [%d,%d] ours=%.6e ref=%.6e diff=%.3e\n" i j o r d)
        (top_diffs ours refm)
  in
  gate "embeddings" (List.nth states 0) (List.nth reference.Hf_loader.hiddens 0) 1e-6 1e-6;
  List.iteri
    (fun i st ->
      if i > 0 && i < n_layers then
        gate (Printf.sprintf "block_%d" (i - 1)) st (List.nth reference.Hf_loader.hiddens i) 1e-3 1e-6)
    states;
  (* hidden_states[-1] is post-ln_f: compare against our final-LN output. *)
  let b11_lnf = Norm.layernorm_forward model.Gpt.final_ln (List.nth states n_layers) in
  gate "block_11_lnf" b11_lnf (List.nth reference.Hf_loader.hiddens n_layers) 1e-3 1e-5;
  gate "logits" logits reference.Hf_loader.logits 1e-4 1e-5;

  if !failures = 0 then Printf.printf "PARITY OK\n" else Printf.printf "PARITY FAILED (%d breaches)\n" !failures;
  exit (if !failures = 0 then 0 else 1)
