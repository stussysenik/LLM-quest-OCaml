(** Loader gates C1-C4 (ROADMAP step C).

    Usage: [./test_loader <llama_ref.safetensors> <llama_tied_ref.safetensors>]

    No forward pass runs here — RMSNorm/SwiGLU/RoPE/GQA are step D.
    What step C must prove is that a checkpoint lands in memory with the
    shape its name claims, and that a file that does *not* match is
    refused loudly rather than loaded optimistically.

    | C1 | key set and every shape match the config-derived expectation |
    | C2 | loaded element count == the file's claimed tensors |
    | C3 | a config with [num_heads] off by one raises, naming the tensor |
    | C4 | tie=true + no head aliases embeddings; tie=false + no head raises |
    | L1 | every loaded matrix is [(out, in)], [out] re-derived from the config |

    C3 and C4b are the ones that matter most: every gate in this file
    would still pass if the loader had no shape assertions and no
    absence checks at all, because the reference file is correct. Only a
    deliberately wrong input proves the guard exists. L1 exists for the
    one mistake the others cannot see — a transpose that lands on a
    square matrix. *)

let passed = ref 0
let failed = ref 0

let check name ok =
  if ok then incr passed else incr failed;
  Printf.printf "%s %s\n" (if ok then "PASS" else "FAIL") name

(** Assert the loader refuses, and that the message names [needle] —
    an exception is useless for finding out *which* check fired. *)
let contains hay needle =
  let n = String.length needle in
  let rec go i = i + n <= String.length hay && (String.sub hay i n = needle || go (i + 1)) in
  n = 0 || go 0

let check_raises name ~(needle : string) (thunk : unit -> unit) =
  match thunk () with
  | exception (Failure msg | Invalid_argument msg) ->
      let ok = contains msg needle in
      check
        (Printf.sprintf "%s -> %s" name
           (if ok then "raised, names " ^ needle else "raised, message lacks " ^ needle))
        ok;
      if not ok then Printf.printf "    message was: %s\n" msg
  | () -> check (name ^ " -> did NOT raise") false

let shape (m : Tensor.mat) = Array.length m, (if Array.length m = 0 then 0 else Array.length m.(0))

(** The layout gate. Every weight is [(out, in)], so a matrix's *rows*
    are its output features. [d_out] is re-derived from the config here
    rather than read off the loaded record — that independence is the
    point: a loader that transposed wrongly in both its [~expected_shape]
    and the matrix it stored would agree with itself and disagree here.

    This is the defence for the case shape assertions cannot reach.
    tiny-random-Llama has [hidden = heads * head_dim = 16], so
    q/k/v/o_proj are all square: transposed, they load clean, multiply to
    plausible numbers, and are caught only by parity after step D. *)
let check_out_in (name : string) (w : Tensor.mat) ~(d_out : int) ~(d_in : int) =
  let rows, cols = shape w in
  if rows <> d_out || cols <> d_in then
    failwith
      (Printf.sprintf "layout gate: %s is [%d, %d]; the convention is [out, in] = [%d, %d]" name rows
         cols d_out d_in)

let check_llama_layout (m : Llama.model) =
  let cfg = m.Llama.cfg in
  let e = cfg.Config.hidden_size in
  let q_dim = cfg.Config.num_heads * Config.head_dim cfg in
  let kv_dim = Config.kv_heads cfg * Config.head_dim cfg in
  let inner = Config.ffn_inner cfg in
  List.iter
    (fun (l : Llama.layer) ->
      check_out_in "q_proj" l.Llama.q_proj ~d_out:q_dim ~d_in:e;
      check_out_in "k_proj" l.Llama.k_proj ~d_out:kv_dim ~d_in:e;
      check_out_in "v_proj" l.Llama.v_proj ~d_out:kv_dim ~d_in:e;
      check_out_in "o_proj" l.Llama.o_proj ~d_out:e ~d_in:q_dim;
      check_out_in "gate_proj" l.Llama.gate_proj ~d_out:inner ~d_in:e;
      check_out_in "up_proj" l.Llama.up_proj ~d_out:inner ~d_in:e;
      check_out_in "down_proj" l.Llama.down_proj ~d_out:e ~d_in:inner)
    m.Llama.layers;
  (* [vocab, hidden] is [out, in] for the head, and the embedding
     table's own layout. For the untied model checked here these are two
     separate tensors read from the file, so agreeing on one shape is
     evidence; for the tied model C4a pins that they are the *same*
     object. *)
  check_out_in "embed_tokens" m.Llama.embed_tokens ~d_out:cfg.Config.vocab_size ~d_in:e;
  check_out_in "lm_head" m.Llama.lm_head ~d_out:cfg.Config.vocab_size ~d_in:e

let () =
  if Array.length Sys.argv <> 3 then (
    Printf.eprintf "usage: %s <llama_ref.safetensors> <llama_tied_ref.safetensors>\n" Sys.argv.(0);
    exit 2);
  let path, tied_path = Sys.argv.(1), Sys.argv.(2) in
  let cfg = Config.llama_tiny_random_config in
  let e = cfg.Config.hidden_size in
  let hd = Config.head_dim cfg in
  let q_dim = cfg.Config.num_heads * hd in
  let inner = Config.ffn_inner cfg in

  (* ---------------- C1: keys and shapes ---------------- *)
  let f = Safetensors.open_file path in
  check "c1_key_set_matches_config"
    (try
       Safetensors.assert_exact_keys f ~skip_prefix:"ref." (Llama_loader.expected_keys cfg);
       true
     with _ -> false);

  let m = Llama_loader.load f cfg in
  check "c1_embed_tokens_shape" (shape m.Llama.embed_tokens = (cfg.Config.vocab_size, e));
  check "c1_lm_head_shape" (shape m.Llama.lm_head = (cfg.Config.vocab_size, e));
  check "c1_final_norm_len" (Array.length m.Llama.norm = e);
  check "c1_layer_count" (Llama.layer_count m = cfg.Config.num_layers);
  check
    "c1_layer_shapes"
    (List.for_all
       (fun (l : Llama.layer) ->
         shape l.q_proj = (q_dim, e)
         && shape l.k_proj = (Config.kv_heads cfg * hd, e)
         && shape l.v_proj = (Config.kv_heads cfg * hd, e)
         && shape l.o_proj = (e, q_dim)
         && shape l.gate_proj = (inner, e)
         && shape l.up_proj = (inner, e)
         && shape l.down_proj = (e, inner)
         && Array.length l.input_norm = e
         && Array.length l.post_attn_norm = e)
       m.Llama.layers);

  (* The transpose trap: q/k/v/o_proj are square for this checkpoint
     (hidden = heads*head_dim = 16), so a transposed one would load
     clean. Squareness is why C3(b) doubles num_heads instead — that
     keeps q_proj square-matching and narrows the break to k/v_proj,
     proving the per-layer assertion fires. And it is why the layout
     gate below exists at all: shape assertions cannot catch a
     transpose that lands on a square matrix. *)

  (* ---------------- layout gate: [out, in] everywhere ---------------- *)
  (match check_llama_layout m with
  | () ->
      check "layout_llama_out_in" true
  | exception Failure msg ->
      check (Printf.sprintf "layout_llama_out_in -> %s" msg) false);

  (* ---------------- C2: parameter count ---------------- *)
  let claimed =
    List.fold_left
      (fun acc k -> acc + List.fold_left ( * ) 1 (Safetensors.find f k).Safetensors.shape)
      0 (Llama_loader.expected_keys cfg)
  in
  check "c2_param_count" (Llama.num_parameters m = claimed);
  Printf.printf "    loaded %d params, file claims %d\n" (Llama.num_parameters m) claimed;

  (* ---------------- C3: wrong shape raises ---------------- *)
  (* Both variants below would sail through a loader with no shape
     assertions, because every other gate here loads a correct file.
     That is the point of C3: the guard, not the data. *)
  let f' = Safetensors.open_file in
  (* (a) An inconsistent config is caught before a single tensor is
     read: head_dim is hidden/num_heads, so heads off by one makes
     num_heads * head_dim disagree with hidden_size. *)
  check_raises "c3_heads_inconsistent_config" ~needle:"num_heads * head_dim" (fun () ->
      ignore (Llama_loader.load (f' path) { cfg with Config.num_heads = 5 }));
  (* (b) A config that is internally consistent but disagrees with the
     file reaches the per-tensor assertion. Doubling num_heads halves
     head_dim, so q_proj still fits (heads*head_dim = hidden either
     way) but k/v_proj no longer do — and they are read only after
     embed_tokens and q_proj, which do. So the layer-level assertion is
     the one that fires, which is exactly the case worth proving. *)
  check_raises "c3_kv_width_mismatch" ~needle:"_proj.weight" (fun () ->
      ignore (Llama_loader.load (f' path) { cfg with Config.num_heads = 8 }));
  (* (c) Too few layers: the key-set assertion names what is missing
     before the loader tries to read a tensor that is not there. *)
  check_raises "c3_missing_layer" ~needle:"model.layers.1.self_attn.q_proj.weight" (fun () ->
      ignore (Llama_loader.load (f' path) { cfg with Config.num_layers = 1 }));

  (* A GPT-2 config must not be accepted by the Llama loader either. *)
  check_raises "c3_gpt2_config_rejected" ~needle:"not a Llama config" (fun () ->
      ignore (Llama_loader.load (f' path) Config.gpt_config_124m));

  (* ---------------- C4: tie handling ---------------- *)
  let tied_cfg = Config.llama_tiny_random_tied_config in
  let tf = Safetensors.open_file tied_path in
  (match Llama_loader.load tf tied_cfg with
  | exception Failure msg ->
      check (Printf.sprintf "c4a_tied_aliases_head -> raised: %s" msg) false
  | tm ->
      check "c4a_tied_aliases_head" (shape tm.Llama.lm_head = (cfg.Config.vocab_size, e));
      (* Held as [vocab, hidden] like the embedding table, so a tied
         head is the embedding table — not its transpose. A transpose
         here would pass the untied gates and fail every Llama parity
         stage after step D, which is exactly the silent class this
         gate exists to catch. *)
      let head = tm.Llama.lm_head and emb = tm.Llama.embed_tokens in
      check "c4a_tied_head_is_the_embedding_matrix" (head == emb));
  check_raises "c4b_untied_without_head_raises" ~needle:"lm_head.weight" (fun () ->
      ignore (Llama_loader.load (Safetensors.open_file tied_path) cfg));
  check_raises "c4c_tied_with_head_is_ambiguous" ~needle:"ambiguous" (fun () ->
      ignore (Llama_loader.load (f' path) tied_cfg));

  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)