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

  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)
