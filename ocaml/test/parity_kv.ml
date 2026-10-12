(** Past-KV cross-check (Gate K5+1): OCaml cache after prefill vs
    PyTorch `past_key_values` on the same 4-token prefix.

    This proves the cache holds the exact tensors PyTorch holds — not
    just that generation converges. fp32-vs-fp64 summation noise on
    K/V (magnitudes O(1-10)) lands ~1e-6, so dual gates abs=1e-3 (with
    headroom) and rel=1e-5.

    Usage: [./parity_kv <gpt2_weights.safetensors> <gpt2_kv_ref.safetensors>]
    (second file written by [ocaml/parity/export_kv.py]). *)

let max_abs_diff (a : Tensor.mat) (b : Tensor.mat) : float =
  let m = ref 0.0 in
  Array.iter2
    (fun ra rb -> Array.iter2 (fun x y -> m := max !m (abs_float (x -. y))) ra rb)
    a b;
  !m

let mat_max_abs (m : Tensor.mat) : float =
  let v = ref 0.0 in
  Array.iter (Array.iter (fun x -> v := max !v (abs_float x))) m;
  !v

let () =
  if Array.length Sys.argv <> 3 then (
    Printf.eprintf "usage: %s <weights.safetensors> <kv_ref.safetensors>\n" Sys.argv.(0);
    exit 2);
  let model = Hf_loader.load_gpt2 Sys.argv.(1) in
  let n_layers = List.length model.Gpt.blocks in
  let n_heads = model.Gpt.cfg.Config.n_heads in
  let kvf = Safetensors.open_file Sys.argv.(2) in
  let ids =
    Array.map Int64.to_int (Safetensors.i64_array_of kvf "kv.input_ids")
  in
  Printf.printf "prefix ids: [%s]\n" (String.concat ";" (List.map string_of_int (Array.to_list ids)));
  let cache = Kv_cache.create ~num_layers:n_layers in
  ignore (Gpt.prefill_cached model cache ids);
  let failures = ref 0 in
  for li = 0 to n_layers - 1 do
    (match Kv_cache.peek cache li with
    | None ->
        Printf.printf "layer %d: EMPTY CACHE\n" li;
        incr failures
    | Some kv ->
        for h = 0 to n_heads - 1 do
          let rk = Safetensors.mat_of_head kvf (Printf.sprintf "pkv.k.%d" li) ~head:h ~heads:n_heads in
          let rv = Safetensors.mat_of_head kvf (Printf.sprintf "pkv.v.%d" li) ~head:h ~heads:n_heads in
          let dk = max_abs_diff kv.Kv_cache.keys.(h) rk in
          let dv = max_abs_diff kv.Kv_cache.values.(h) rv in
          let ok =
            dk <= 1e-3
            && dv <= 1e-3
            && dk /. max (mat_max_abs rk) 1e-30 <= 1e-5
            && dv /. max (mat_max_abs rv) 1e-30 <= 1e-5
          in
          if not ok then (
            Printf.printf "layer %d head %d: k_diff=%.3e v_diff=%.3e BREACH\n" li h dk dv;
            incr failures)
        done);
    Printf.printf "layer %d: %s\n" li (if !failures = 0 then "OK" else "see breaches above")
  done;
  if !failures = 0 then Printf.printf "KV PARITY OK\n" else Printf.printf "KV PARITY FAILED\n";
  exit (if !failures = 0 then 0 else 1)
