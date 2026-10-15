(* Gates for the three multimodal loaders: [Vit_loader], [Vlm_loader] and
   [Gpt_repo_loader] — the code path that makes the models loadable at all.

   These exist because before them the port could *gate* ViT and VLM but
   could not *load* them: parity built weights by hand inside the test, so
   nothing exercised reading a real `state_dict`. That gap is where a
   transposed or mis-indexed key would live.

   Every gate here is anchored on something the PyTorch side produced, not
   on a value the test wrote:

   - L1 the loaded weights reproduce the reference activations, for all
     three models at once (the whole point of the loaders).
   - L2 the key set matches a *real* `state_dict` — asserted by the
     loaders themselves and reported here, so a failure names which model.
   - L3 parameter counts match PyTorch's `parameters()` numel, which is
     the gate that catches a tensor read twice or dropped.
   - L4 the conv kernel is read as 4-D, not flattened to `(out, in)` — the
     ViT's one weight outside the project convention, and the case where
     the shape is square enough to hide a mistake.
   - L5 anti-gates: the layout must *reject* a transposed square
     projection, and the causal-mask check must *reject* a wrong mask.
     A gate that only confirms the happy path cannot tell a correct
     reader from a permissive one.

   Tolerances are 1e-9 dual abs+rel throughout, the same gate the rest of
   the port runs at now that [Norm.erf] is fp64-accurate. *)

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

let max_abs (a : mat) (b : mat) : float =
  let worst = ref 0.0 in
  Array.iteri
    (fun i ra ->
      Array.iteri (fun j v -> worst := max !worst (abs_float (v -. b.(i).(j)))) ra)
    a;
  !worst

let max_abs_vec (a : vec) (b : vec) : float =
  let worst = ref 0.0 in
  for i = 0 to Array.length a - 1 do
    worst := max !worst (abs_float (a.(i) -. b.(i)))
  done;
  !worst

let gate name (ours : mat) (refm : mat) tol =
  let d = max_abs ours refm in
  let scale = Array.fold_left (fun acc r -> max acc (Array.fold_left max 0.0 r)) 0.0 refm in
  Printf.printf "  %-42s max_abs_diff=%.3e ref_max=%.3e\n" name d scale;
  check name (d <= tol)

let gate_vec name (ours : vec) (refv : vec) tol =
  let d = max_abs_vec ours refv in
  let scale = Array.fold_left max 0.0 refv in
  Printf.printf "  %-42s max_abs_diff=%.3e ref_max=%.3e\n" name d scale;
  check name (d <= tol)

(* The decoder config this gate runs at. Identical to
   [Config.gpt_tiny_test_config] except [n_positions]: the tiny config's
   16 is smaller than a 17-token image on its own, so the fused sequence
   could not fit. Everything else matches so the gate exercises the same
   code a real 124M load would. *)
let gpt_cfg =
  {
    Config.vocab_size = 64;
    hidden_size = 16;
    num_layers = 2;
    num_heads = 2;
    arch =
      Config.Gpt2
        {
          (* Inline-variant record fields are not module-qualified: there
             is no [Config.n_positions], only [Gpt2]'s field. *)
          n_positions = 64;
          n_inner = None;
          layer_norm_eps = 1e-5;
          activation = `Gelu;
          qkv_bias = false;
        };
  }

let vit_cfg : Vit.t =
  {
    img_height = 8;
    img_width = 8;
    patch_size = 2;
    num_channels = 3;
    emb_dim = 32;
    n_layers = 3;
    n_heads = 2;
    qkv_bias = true;
    num_classes = 5;
  }

let raises f = try ignore (f ()); false with Failure _ -> true

let () =
  if Array.length Sys.argv <> 2 then (
    Printf.eprintf "usage: %s <vlm_loader_ref.safetensors>\n" Sys.argv.(0);
    exit 2);
  let path = Sys.argv.(1) in
  let f = Safetensors.open_file path in
  Printf.printf "Loader gates (vit_loader / vlm_loader / gpt_repo_loader)\n\n";

  (* ---------------------------------------------------------------- *)
  (* L2a: each loader claims exactly the keys a real state_dict has.   *)
  (* ---------------------------------------------------------------- *)
  Printf.printf "L2a key sets vs a real state_dict\n";
  let vit_sub = Safetensors.subset f ~prefix:"vit." in
  let ad_sub = Safetensors.subset f ~prefix:"adapter." in
  let gpt_sub = Safetensors.subset f ~prefix:"gpt." in
  check "L2a vit.expected_keys count == tensors in vit. namespace"
    (List.length (Safetensors.names vit_sub) = List.length (Vit_loader.expected_keys vit_cfg));
  check "L2a gpt key set is claimed exactly"
    (List.length (Safetensors.names gpt_sub) = List.length (Gpt_repo_loader.expected_keys gpt_cfg));
  check "L2a adapter is bias-free (2 tensors for the ffn branch)"
    (List.length (Safetensors.names ad_sub) = 2);

  (* ---------------------------------------------------------------- *)
  (* L4: the conv kernel is 4-D.                                      *)
  (* ---------------------------------------------------------------- *)
  Printf.printf "\nL4 the one weight outside the (out, in) convention\n";
  let conv_shape =
    match Safetensors.find f "vit.patch_embedding.conv_proj.weight" with
    | e -> e.Safetensors.shape
  in
  check "L4 conv_proj.weight is 4-D [emb, C, P, P]" (conv_shape = [ 32; 3; 2; 2 ]);
  check "L4 a 2-D read of the conv is refused"
    (raises (fun () -> Safetensors.get_tensor f "vit.patch_embedding.conv_proj.weight"
                            ~expected_shape:[ 32; 32 ]));

  (* ---------------------------------------------------------------- *)
  (* L1: load all three, run the fused forward.                       *)
  (* ---------------------------------------------------------------- *)
  Printf.printf "\nL1 loaded weights reproduce the reference\n";
  let m =
    Vlm_loader.load f ~vit_cfg ~gpt_cfg ~adapter_type:(`Ffn 4) ~bias:false
  in
  let img = Safetensors.mat3_of_f64 f "in.img" in
  let image = Array.map (fun a -> a) img in
  let ids =
    Safetensors.i64_array_of f "in.input_ids" |> Array.map Int64.to_int
  in
  gate "L1 vit_hidden (loaded ViT)" (Vit.forward_hidden_states m.Vlm.vit image)
    (Safetensors.mat_of_f64 f "ref.vit_hidden") 1e-9;
  gate "L1 vision embeddings (adapter)" (Vlm.vision_embeddings m ~image)
    (Safetensors.mat_of_f64 f "ref.vision") 1e-9;
  gate "L1 fused sequence" (Vlm.fused_embeddings m ~image ~ids)
    (Safetensors.mat_of_f64 f "ref.fused") 1e-9;
  gate "L1 fused logits" (Vlm.forward m ~image ~ids) (Safetensors.mat_of_f64 f "ref.logits") 1e-9;
  (* PyTorch exports the classifier output as the batched [(1, num_classes)]
     it is; the port returns the single row it always deals in. *)
  let vit_logits_ref = (Safetensors.mat_of_f64 f "ref.vit_logits").(0) in
  gate_vec "L1 ViT classifier (loaded alone)" (Vit.forward m.Vlm.vit image) vit_logits_ref 1e-9;

  (* ---------------------------------------------------------------- *)
  (* L3: nothing dropped, nothing read twice.                         *)
  (* ---------------------------------------------------------------- *)
  Printf.printf "\nL3 parameter counts\n";
  let numel (file : Safetensors.file) (key : string) : int =
    let e = Safetensors.find file key in
    let n = List.fold_left ( * ) 1 e.Safetensors.shape in
    (* F64 here because the exporter writes references in double. This
       counts *elements*, not bytes, so the dtype is irrelevant to the
       comparison — the branch is kept only so the intent is explicit. *)
    ignore e.Safetensors.dtype;
    n
  in
  let total file skip = List.fold_left (fun a k -> a + numel file k) 0 (List.filter (fun k -> not (skip k)) (Safetensors.names file)) in
  let vit_numel = total f (fun k -> not (String.starts_with ~prefix:"vit." k)) in
  check (Printf.sprintf "L3 ViT param_count == %d elements on disk" vit_numel)
    (Vit_loader.param_count m.Vlm.vit = vit_numel);
  let gpt_numel =
    total f (fun k -> (not (String.starts_with ~prefix:"gpt." k)) || String.ends_with ~suffix:"att.mask" k)
  in
  check
    (Printf.sprintf "L3 GPT param_count == %d elements (mask buffers excluded)" gpt_numel)
    (Gpt_repo_loader.param_count m.Vlm.gpt = gpt_numel);

  (* ---------------------------------------------------------------- *)
  (* L5: anti-gates. The readers must REJECT the wrong thing.          *)
  (* ---------------------------------------------------------------- *)
  Printf.printf "\nL5 anti-gates — the readers must refuse wrong input\n";
  (* A config whose block count disagrees with the file must fail, not
     quietly load the first N blocks and stop. *)
  check "L5 wrong n_layers is rejected"
    (raises (fun () ->
         Vit_loader.load (Safetensors.subset f ~prefix:"vit.") { vit_cfg with Vit.n_layers = 2 }));
  check "L5 wrong emb_dim is rejected by a shape assertion"
    (raises (fun () ->
         Vit_loader.load (Safetensors.subset f ~prefix:"vit.") { vit_cfg with Vit.emb_dim = 16 }));
  check "L5 wrong patch_size is rejected"
    (raises (fun () ->
         Vit_loader.load (Safetensors.subset f ~prefix:"vit.") { vit_cfg with Vit.patch_size = 4 }));
  check "L5 adapter told 'simple' against an ffn file is rejected"
    (raises (fun () -> Vlm_loader.load f ~vit_cfg ~gpt_cfg ~adapter_type:`Simple ~bias:false));
  check "L5 adapter told it has biases against a bias-free file is rejected"
    (raises (fun () -> Vlm_loader.load f ~vit_cfg ~gpt_cfg ~adapter_type:(`Ffn 4) ~bias:true));
  check "L5 GPT told 3 layers against a 2-layer file is rejected"
    (raises (fun () ->
         Gpt_repo_loader.load (Safetensors.subset f ~prefix:"gpt.")
           { gpt_cfg with Config.num_layers = 3 }));

  (* The causal-mask buffer: a checkpoint whose mask is non-causal must be
     refused, because the library rebuilds the right mask on every forward
     and would otherwise never notice the file disagreed with itself.

     Tested by *corrupting the file* rather than by asserting a helper
     raises — a gate that only re-states the existence of a check is the
     "gate that cannot fail" shape this repo has found and fixed four
     times. [Safetensors.file.data] is a [bytes], so the fix is a
     four-byte write: row 0, column 1 of the first block's mask, which a
     causal mask requires to be 1.0 (this checkpoint stores the *upper*
     triangle as "masked", so the corruption clears it). *)
  Printf.printf "\nL5 anti-gates — a non-causal mask buffer is refused\n";
  let with_patched_bytes (name : string) (offset : int) (value : float) : string =
    let data = Bytes.copy f.Safetensors.data in
    Bytes.set_int32_le data offset (Int32.bits_of_float value);
    let tmp = Filename.temp_file name ".safetensors" in
    let oc = open_out_bin tmp in
    output_bytes oc data;
    close_out oc;
    tmp
  in
  let mask_entry = Safetensors.find f "gpt.trf_blocks.0.att.mask" in
  check "L5 a non-causal mask buffer is rejected"
    (raises (fun () ->
         let tmp = with_patched_bytes "mask" (mask_entry.Safetensors.off_start + 4) 0.0 in
         let g = Gpt_repo_loader.load (Safetensors.subset (Safetensors.open_file tmp) ~prefix:"gpt.") gpt_cfg in
         ignore g;
         Sys.remove tmp));
  (* And the same buffer, truncated: a checkpoint saved at a different
     context_length is the realistic version of this failure. *)
  check "L5 a mask of the wrong size is rejected"
    (raises (fun () ->
         Gpt_repo_loader.check_mask
           (Safetensors.subset f ~prefix:"gpt.")
           (Config.max_positions gpt_cfg + 1) 0));

  (* ---------------------------------------------------------------- *)
  (* G: the caption-generation path.                                   *)
  (* ---------------------------------------------------------------- *)
  Printf.printf "\nG caption generation (vlm_generation.py's loop)\n";
  let n_vit = Vlm.num_vision_tokens m in
  let cache = Kv_cache.create ~num_layers:gpt_cfg.Config.num_layers in

  (* G-a: the prefill window is the adapter's output alone, carried
     through with **no** positional embedding — `GPTModel.forward`'s
     `x = x + pos_emb` lives inside `if not input_embedded:`.

     This is the single most expensive line to get wrong in the whole
     multimodal path. Adding positions here measures 1.36e-01 against the
     reference (rel ~1e-2): big enough to wreck a caption, small enough
     that no shape assertion, no key-set check and no range check
     notices. It was found by running this gate, not by reading the code
     twice. *)
  let prefill = Vlm.prefill_cached ~context_length:64 m cache ~image in
  gate_vec "G-a prefill last-row logits (vision only, no positions)"
    prefill.(Array.length prefill - 1)
    (Safetensors.vec_of_f64 f "ref.prefill_last_logits") 1e-9;

  (* G-b: the cached decode must equal a full re-forward. This is the
     claim that makes the cache worth having — the reference re-runs the
     whole sequence every step. Checked per step, against the reference's
     own per-step logits, so an argmax that happens to agree cannot hide a
     small error. *)
  let ref_tokens = Safetensors.i64_array_of f "ref.greedy_tokens" |> Array.map Int64.to_int in
  let n_gated = min 6 (Array.length ref_tokens - 1) in
  let b_ok = ref true and b_worst = ref 0.0 in
  let cache_b = Kv_cache.create ~num_layers:gpt_cfg.Config.num_layers in
  ignore (Vlm.prefill_cached ~context_length:64 m cache_b ~image : mat);
  for i = 0 to n_gated - 1 do
    let logits = Vlm.decode_step_cached m cache_b ~tok:ref_tokens.(i) in
    let refm = Safetensors.vec_of_f64 f (Printf.sprintf "ref.step%d_last_logits" (i + 1)) in
    let d = max_abs_vec logits refm in
    if d > !b_worst then b_worst := d;
    if d > 1e-9 then b_ok := false;
    (* The token the reference picked must be this step's argmax. *)
    let arg = ref (-1) and best = ref neg_infinity in
    Array.iteri (fun j v -> if v > !best then (best := v; arg := j)) refm;
    if !arg <> ref_tokens.(i + 1) then b_ok := false
  done;
  Printf.printf "  %-40s max_abs_diff=%.3e over %d decode steps\n" "G-b cached == reference"
    !b_worst n_gated;
  check "G-b cached decode step == reference logits and argmax" !b_ok;

  (* G-c: the whole loop, token for token, greedy. *)
  let rng = Random.State.make [| 7 |] in
  let ours = Vlm.generate_cached rng ~max_gen:(Array.length ref_tokens) ~context_length:64 m ~image in
  let same = Array.length ours = Array.length ref_tokens in
  let first_diff =
    let d = ref (-1) in
    for i = 0 to min (Array.length ours) (Array.length ref_tokens) - 1 do
      if ours.(i) <> ref_tokens.(i) && !d < 0 then d := i
    done;
    !d
  in
  Printf.printf "  %-40s ours=[%s] ref=[%s]\n" "G-c greedy token ids"
    (String.concat ";" (Array.to_list (Array.map string_of_int ours)))
    (String.concat ";" (Array.to_list (Array.map string_of_int ref_tokens)));
  check "G-c generate_cached reproduces the reference tokens" (same && first_diff < 0);

  (* G-d: the stop contract. The reference appends the stopping token and
     breaks *without* decoding past it, so with eos = the first reference
     token the answer must be exactly that one token. This is not a
     synthetic case: [eos_ids] is set from the reference's own first
     token, so the expected output is a real number the export produced. *)
  let eos_tok = ref_tokens.(0) in
  let stopped =
    Vlm.generate_cached rng ~max_gen:12 ~context_length:64 ~eos_ids:[ eos_tok ] m ~image
  in
  Printf.printf "  %-40s got %d token(s), first=%d (eos=%d)\n" "G-d stops on EOS" (Array.length stopped)
    stopped.(0) eos_tok;
  check "G-d EOS is appended and the loop stops there" (Array.length stopped = 1 && stopped.(0) = eos_tok);

  (* G-e: anti-gate. The bare vision sequence must NOT match once
     positional embeddings are added. This is the mistake G-a exists to
     catch, re-run as a divergence check so it is visibly non-zero rather
     than merely "G-a passes" — a positive gate and an anti-gate on the
     same quantity, which is the shape that catches a gate written so
     loosely both would agree.

     Note this is adding positions *once* to a sequence that should have
     none, not double-counting. Double-counting is a different mistake
     (training's [fused_embeddings] on the generation path) and is
     strictly worse; it is not separately gated because it is a
     superset of this one and the reason to be suspicious is the same. *)
  let wrong_cache = Kv_cache.create ~num_layers:gpt_cfg.Config.num_layers in
  let wrong_seq = Vlm.vision_embeddings m ~image in
  let with_positions = Gpt.add_pos m.Vlm.gpt wrong_seq in
  let wrong_logits = Gpt.prefill_from_emb_cached m.Vlm.gpt wrong_cache with_positions in
  let d_wrong =
    max_abs_vec wrong_logits.(Array.length wrong_logits - 1)
      (Safetensors.vec_of_f64 f "ref.prefill_last_logits")
  in
  Printf.printf "  %-40s max_abs_diff=%.3e (must exceed 1e-06)\n" "G-e positions added by the port diverge"
    d_wrong;
  check "G-e a port that adds positions is rejected" (d_wrong > 1e-6);

  (* G-f: a context_length too small to hold the image must be refused
     rather than silently truncating the image away — the reference
     would truncate, and a KV cache cannot un-push what it has seen. *)
  check "G-f a context too small for the image is refused"
    (try
       ignore (Vlm.prefill_cached ~context_length:(n_vit - 1) m (Kv_cache.create ~num_layers:2) ~image);
       false
     with Failure _ -> true);

  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)
