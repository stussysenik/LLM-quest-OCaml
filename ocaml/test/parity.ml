(** Parity harness: OCaml vs PyTorch, parameterized by model family.

    Usage: [./parity_check [--family gpt2|llama] <weights_with_ref.safetensors>]
    (file written by [ocaml/parity/export_reference.py]).

    Gates (from [ocaml/ROADMAP.md]):
    - embeddings (hidden.0) match to ~1e-6  → weight loader correct
    - block outputs (hidden.1..L) to ~1e-5  → architecture correct
    - post-final-norm state to ~1e-5
    - final logits to ~1e-4                 → fp32 accumulation over 12 layers

    [run] below is the whole harness: family-independent by construction.
    A family contributes a [runnable] — the family's concrete model type
    erased behind five operations — plus a loader. Adding DeepSeek is a
    new record here and a new [export_deepseek.py]; [run] does not change.

    Two calibration lessons from the first run against real weights:
    - GPT-2 has outlier dims (~3000); fp32-vs-fp64 summation noise on
      those is ~4e-4 absolute (~1e-7 relative), so block gates use a
      dual abs+rel criterion rather than a flat 1e-5.
    - transformers 5.x appends the POST-ln_f state as hidden_states[-1]
      (verified: hidden.12 @ lm_head.T == logits, and its std is ~5,
      not ~80 like the raw outlier-dominated states). So the last block
      is compared against our final-norm output, not the raw block
      output; the export normalizes the same slot for every family, so
      a convention difference surfaces at export, not at parity time.

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

(* ------------------------------------------------------------------ *)
(* The family boundary                                                  *)
(* ------------------------------------------------------------------ *)

(** Everything the stage loop needs from a loaded model, with the
    family's concrete type erased. Nothing here is family-specific:
    both families embed, run [n_layers] pre-LN blocks, apply one final
    norm, and project to vocab. *)
type runnable = {
  n_layers : int;
  states : int array -> Tensor.mat list; (** [embeddings; block_0; ...] *)
  final_norm : Tensor.mat -> Tensor.mat;
  logits : int array -> Tensor.mat;
  final_norm_label : string; (** stage name: ["block_11_lnf"], ["block_1_rmsnorm"] *)
}

type family = { name : string; load : string -> runnable }

(** The layout gate: every weight a loader hands the kernels is
    [(out, in)], so a matrix's *rows* are its output features. The
    expected [out] is re-derived from the config here rather than read
    back off the loaded record — that independence is the entire point. A
    loader that transposed wrongly in both its [~expected_shape] and the
    matrix it stored would agree with itself and disagree with this.

    Shape assertions alone cannot cover the case that matters. For
    [gpt2], and for tiny-random-Llama, [hidden = heads * head_dim], so
    q/k/v/o_proj are all square: a transposed weight multiplies to
    plausible numbers and only parity notices, long after the mistake. *)
let check_out_in (name : string) (w : Tensor.mat) ~(d_out : int) ~(d_in : int) =
  let rows = Array.length w in
  let cols = if rows = 0 then 0 else Array.length w.(0) in
  if rows <> d_out || cols <> d_in then
    failwith
      (Printf.sprintf "layout gate: %s is [%d, %d]; the convention is [out, in] = [%d, %d]" name rows
         cols d_out d_in)

let check_gpt2_layout (m : Gpt.t) =
  let cfg = m.Gpt.cfg in
  let e = cfg.Config.hidden_size in
  let q_dim = cfg.Config.num_heads * Config.head_dim cfg in
  let inner = Config.ffn_inner cfg in
  let lin name (l : Attention.linear) ~d_out ~d_in = check_out_in name l.Attention.w ~d_out ~d_in in
  List.iter
    (fun (b : Block.block) ->
      let att = b.Block.att in
      lin "attn.w_q" att.Attention.w_q ~d_out:q_dim ~d_in:e;
      lin "attn.w_k" att.Attention.w_k ~d_out:q_dim ~d_in:e;
      lin "attn.w_v" att.Attention.w_v ~d_out:q_dim ~d_in:e;
      lin "attn.c_proj" att.Attention.out_proj ~d_out:e ~d_in:q_dim;
      lin "mlp.c_fc" b.Block.ffn.Block.l1 ~d_out:inner ~d_in:e;
      (* [mlp.c_proj] reads the FFN's width, not [hidden] — the one
         projection whose input is not the residual stream. *)
      lin "mlp.c_proj" b.Block.ffn.Block.l2 ~d_out:e ~d_in:inner)
    m.Gpt.blocks;
  check_out_in "lm_head" m.Gpt.lm_head.Attention.w ~d_out:cfg.Config.vocab_size ~d_in:e;
  (* The embedding tables are not linears — they are looked up, never
     multiplied — so they are the one place a layout convention could be
     got wrong without [linear_fwd] caring. Both are [vocab, hidden] and
     [max_positions, hidden], which is [out, in] on the reading this
     project uses for the head as well. *)
  check_out_in "transformer.wte" m.Gpt.tok_emb ~d_out:cfg.Config.vocab_size ~d_in:e;
  check_out_in "transformer.wpe" m.Gpt.pos_emb ~d_out:(Config.max_positions cfg) ~d_in:e

let gpt2 : family =
  {
    name = "gpt2";
    load =
      (fun path ->
        let m = Gpt2_loader.load (Safetensors.open_file path) Config.gpt_config_124m in
        check_gpt2_layout m;
        Printf.printf "layout gate: gpt2 linear weights are [out, in]\n";
        let n_layers = List.length m.Gpt.blocks in
        {
          n_layers;
          states = (fun ids -> Gpt.forward_states m ids);
          final_norm = (fun x -> Norm.layer_norm m.Gpt.final_ln x);
          logits = (fun ids -> Gpt.forward m ids);
          final_norm_label = Printf.sprintf "block_%d_lnf" (n_layers - 1);
        });
  }

(** The Llama loader landed in ROADMAP step C and does run — but there
    is no forward pass yet: RMSNorm (D1) and SwiGLU (D2) exist and are
    gated in isolation, while RoPE (D3/D4) and GQA (D5) have not.
    Weights load first so that reaching this message means the loader
    worked on the real file; the boundary it reports is the kernels,
    not the parsing. *)
let llama : family =
  {
    name = "llama";
    load =
      (fun path ->
        let _weights = Llama_loader.load (Safetensors.open_file path) Config.llama_tiny_random_config in
        failwith
          "Parity: --family llama loads its weights, but the forward pass needs ROADMAP step D (D3/D4 RoPE, D5 GQA; D1 RMSNorm and D2 SwiGLU are done)");
  }

let families = [ gpt2; llama ]

let family_by_name name =
  match List.find_opt (fun f -> f.name = name) families with
  | Some f -> f
  | None -> failwith (Printf.sprintf "Parity: unknown family %S (have: %s)" name
                       (String.concat ", " (List.map (fun f -> f.name) families)))

(* ------------------------------------------------------------------ *)
(* The reference file                                                   *)
(* ------------------------------------------------------------------ *)

(** The [ref.*] half of the parity file, written by
    [parity/export_reference.py]. Lives here, not in a loader: the
    weights and the reference share one file, but only the harness knows
    that [ref.*] is the reference and not a weight. *)
type reference = { ids : int array; hiddens : Tensor.mat list; logits : Tensor.mat }

let load_reference (path : string) ~n_layers : reference =
  let f = Safetensors.open_file path in
  let ids64 = Safetensors.i64_array_of f "ref.input_ids" in
  let ids = Array.map Int64.to_int ids64 in
  let hiddens =
    List.init (n_layers + 1) (fun i -> Safetensors.mat_of f (Printf.sprintf "ref.hidden.%d" i))
  in
  let logits = Safetensors.mat_of f "ref.logits" in
  { ids; hiddens; logits }

(* ------------------------------------------------------------------ *)
(* The harness                                                          *)
(* ------------------------------------------------------------------ *)

(** The one stage loop. Reads a family, runs the same ladder for it —
    one gate per hidden state the export wrote ([1 + n_layers] of them,
    14 for gpt2), then logits — reports the same way, returns the
    breach count. *)
let run ~(family : family) (path : string) : int =
  let model = family.load path in
  let n_layers = model.n_layers in
  let reference = load_reference path ~n_layers in
  Printf.printf "family: %s\n" family.name;
  Printf.printf "prompt ids: [%s]\n"
    (String.concat ";" (List.map string_of_int (Array.to_list reference.ids)));

  let states = model.states reference.ids in
  let logits = model.logits reference.ids in

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
  gate "embeddings" (List.nth states 0) (List.nth reference.hiddens 0) 1e-6 1e-6;
  List.iteri
    (fun i st ->
      if i > 0 && i < n_layers then
        gate (Printf.sprintf "block_%d" (i - 1)) st (List.nth reference.hiddens i) 1e-3 1e-6)
    states;
  (* ref.hidden.{n_layers} is the post-final-norm state (see header). *)
  let final_normed = model.final_norm (List.nth states n_layers) in
  gate model.final_norm_label final_normed (List.nth reference.hiddens n_layers) 1e-3 1e-5;
  gate "logits" logits reference.logits 1e-4 1e-5;
  !failures

let () =
  let family = ref "gpt2" in
  let positional = ref [] in
  let rec parse i =
    if i >= Array.length Sys.argv then ()
    else if Sys.argv.(i) = "--family" then (
      if i + 1 >= Array.length Sys.argv then (
        Printf.eprintf "parity_check: --family needs a value\n";
        exit 2);
      family := Sys.argv.(i + 1);
      parse (i + 2))
    else (
      positional := Sys.argv.(i) :: !positional;
      parse (i + 1))
  in
  parse 1;
  let files = List.rev !positional in
  if List.length files <> 1 then (
    Printf.eprintf "usage: %s [--family gpt2|llama] <weights_with_ref.safetensors>\n" Sys.argv.(0);
    exit 2);
  let fam = family_by_name !family in
  let failures = run ~family:fam (List.hd files) in
  if failures = 0 then Printf.printf "PARITY OK\n" else Printf.printf "PARITY FAILED (%d breaches)\n" failures;
  exit (if failures = 0 then 0 else 1)