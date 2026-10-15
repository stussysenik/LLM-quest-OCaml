(** ViT gates V-a..V-g (multimodal Part 1).

     Usage: [./test_vit <vit_ref.safetensors>]
     (written by [ocaml/parity/export_vit.py], which instantiates the
     repo's own [ViTModel] with deterministic weights).

     The ViT is GPT's decoder with four changes, and each is gated
     separately because each is a place where reusing a GPT kernel would
     be *almost* right:

     | V-a | patch embedding: conv stride == kernel, [emb, C*P*P] kernel, bias |
     | V-b | patch order is row-major over the (H/P, W/P) grid |
     | V-c | CLS token is prepended at sequence position 0 |
     | V-d | attention is **unmasked** — the decoder form with every position visible must agree with the encoder form |
     | V-e | LayerNorm convention: eps **outside** the sqrt, not HF's inside |
     | V-f | logits come from the CLS token, and the whole stack composes |
     | V-g | GELU is the exact fp64 erf path, gated at 1e-13, not bounded at A&S's 2e-7 |

     V-d and V-e are the two that a GPT-inherited implementation gets
     wrong while still looking right.

     V-d: a ViT block that silently kept the causal mask produces
     perfectly plausible attention — every row is a valid softmax — and a
     self-consistent model. Nothing raises. The gate therefore does not
     just check that the encoder form runs; it checks that the encoder
     form equals the *decoder* form on a sequence short enough that the
     causal mask hides nothing. That turns "we used the unmasked kernel"
     into a numeric claim rather than a claim about the source.

     V-e: ViT's LayerNorm is [(x - mean) / (std + eps)]; HuggingFace's is
     [(x - mean) / sqrt(var + eps)]. ~5e-6 relative apart — invisible in
     training, fatal here. Pinning it needs both a positive gate (the
     outside-the-sqrt form matches) and a negative one (the inside-the-sqrt
     form does not), because a gate that only checks the positive case
     would be satisfied by any normalization.

     Two shapes are gated, for the reason D2 uses two: `small` has 16
     patches (sequence 17) and `odd` has 4 patches with a patch size that
     does not divide the head count the same way (sequence 5). Both
     exercise `pos_embedding`'s length against the sequence it is added to. *)

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

let mat_max_abs (m : Tensor.mat) : float =
  let v = ref 0.0 in
  Array.iter (Array.iter (fun x -> v := max !v (abs_float x))) m;
  !v

let max_abs_diff_mat (a : Tensor.mat) (b : Tensor.mat) : float =
  let rows = Array.length a in
  if rows <> Array.length b then failwith "test_vit: row count mismatch";
  let m = ref 0.0 in
  for i = 0 to rows - 1 do
    if Array.length a.(i) <> Array.length b.(i) then failwith "test_vit: col count mismatch";
    for j = 0 to Array.length a.(i) - 1 do
      m := max !m (abs_float (a.(i).(j) -. b.(i).(j)))
    done
  done;
  !m

let check_diverges name ~min_gap a b =
  let d = max_abs_diff_mat a b in
  let ok = d > min_gap in
  check (Printf.sprintf "%s (max_abs_diff=%.3e, must exceed %.0e)" name d min_gap) ok

let max_abs_diff_vec (a : Tensor.vec) (b : Tensor.vec) : float =
  let m = ref 0.0 in
  for i = 0 to Array.length a - 1 do
    m := max !m (abs_float (a.(i) -. b.(i)))
  done;
  !m

let check_diverges_vec name ~min_gap (a : Tensor.vec) (b : Tensor.vec) =
  let d = max_abs_diff_vec a b in
  let ok = d > min_gap in
  check (Printf.sprintf "%s (max_abs_diff=%.3e, must exceed %.0e)" name d min_gap) ok

let gate name ours refm abs_tol rel_tol =
  let diff = max_abs_diff_mat ours refm in
  let refmag = mat_max_abs refm in
  let rel = diff /. max refmag 1e-30 in
  let ok = diff <= abs_tol && rel <= rel_tol in
  if ok then incr passed else incr failed;
  Printf.printf "%-22s max_abs_diff=%.3e ref_max=%.3e rel=%.3e %s\n" name diff refmag rel
    (if ok then "OK" else "BREACH")

let gate_vec name ours (refv : Tensor.vec) abs_tol rel_tol =
  let diff = max_abs_diff_vec ours refv in
  let refmag = Array.fold_left (fun a v -> max a (abs_float v)) 0.0 refv in
  let rel = diff /. max refmag 1e-30 in
  let ok = diff <= abs_tol && rel <= rel_tol in
  if ok then incr passed else incr failed;
  Printf.printf "%-22s max_abs_diff=%.3e ref_max=%.3e rel=%.3e %s\n" name diff refmag rel
    (if ok then "OK" else "BREACH")

let mk_linear (w : mat) (b : vec) : Attention.linear =
  let d_out = Array.length w in
  let d_in = if d_out = 0 then 0 else Array.length w.(0) in
  { Attention.w; b = Some b; d_in; d_out }

let img_of_f64_3d (f : Safetensors.file) name ~(c : int) ~(h : int) ~(w : int) : Vit.image =
  let img = Safetensors.mat3_of_f64 f name in
  if Array.length img <> c || Array.length img.(0) <> h || Array.length img.(0).(0) <> w then
    failwith (Printf.sprintf "test_vit: %s is not [%d,%d,%d]" name c h w);
  img

let run_case (f : Safetensors.file) (name : string) ~(img_h : int) ~(img_w : int) ~(patch : int)
    ~(chans : int) ~(emb : int) ~(layers : int) ~(heads : int) ~(classes : int) =
  let p = Printf.sprintf "v1.%s" name in
  let g name = Printf.sprintf "%s.%s" p name in
  let n_patch = (img_h * img_w) / (patch * patch) in
  let seq = n_patch + 1 in

  let cfg : Vit.t =
    {
      Vit.img_height = img_h;
      img_width = img_w;
      patch_size = patch;
      num_channels = chans;
      emb_dim = emb;
      n_layers = layers;
      n_heads = heads;
      qkv_bias = true;
      num_classes = classes;
    }
  in

  let v name = Safetensors.vec_of_f64 f (g name) in
  let mv name = Safetensors.mat_of_f64 f (g name) in

  let blocks =
    Array.init layers (fun i ->
        let b = Printf.sprintf "%s.block%d" p i in
        let vn n = Safetensors.vec_of_f64 f (Printf.sprintf "%s.%s" b n) in
        let vm n = Safetensors.mat_of_f64 f (Printf.sprintf "%s.%s" b n) in
        {
          Vit.ln_1 = { Norm.scale = vn "ln1_scale"; shift = vn "ln1_shift"; eps = 1e-5; eps_placement = Norm.Eps_outside_sqrt };
          ln_2 = { Norm.scale = vn "ln2_scale"; shift = vn "ln2_shift"; eps = 1e-5; eps_placement = Norm.Eps_outside_sqrt };
          att =
            {
              Attention.w_q = mk_linear (vm "att_wq_weight") (vn "att_wq_bias");
              w_k = mk_linear (vm "att_wk_weight") (vn "att_wk_bias");
              w_v = mk_linear (vm "att_wv_weight") (vn "att_wv_bias");
              out_proj = mk_linear (vm "att_wo_weight") (vn "att_wo_bias");
              num_heads = heads;
              head_dim = emb / heads;
              d_out = emb;
              ctx_len = seq;
            };
          ffn_l1 = mk_linear (vm "ffn_l1_weight") (vn "ffn_l1_bias");
          ffn_l2 = mk_linear (vm "ffn_l2_weight") (vn "ffn_l2_bias");
        })
  in

  let model : Vit.model =
    {
      Vit.cfg = cfg;
      patch = { conv_weight = mv "conv_weight"; conv_bias = v "conv_bias"; cls_token = v "cls_token" };
      pos_embedding = mv "pos_embed";
      blocks;
      final_ln =
        {
          Norm.scale = v "final_scale";
          shift = v "final_shift";
          eps = 1e-5;
          eps_placement = Norm.Eps_outside_sqrt;
        };
      classifier = mk_linear (mv "classifier_weight") (v "classifier_bias");
    }
  in

  let img = img_of_f64_3d f (g "img") ~c:chans ~h:img_h ~w:img_w in
  let ref_hidden = mv "ref_hidden" in
  let ref_logits = Safetensors.vec_of_f64 f (g "ref_logits") in

  (* ---- V-a: patch embedding ---- *)
  Printf.printf "%s: V-a patch embedding (%d patches)\n" name n_patch;
  let pe = Vit.patch_embedding model img in
  if Array.length pe <> seq then failwith "test_vit: patch embedding sequence length";
  (* ViTModel's first stage is conv -> flatten(2) -> transpose(1,2) ->
     cat(cls); the reference is that stage's own output. *)
  gate (Printf.sprintf "V-a patch embeddings (%s)" name) pe (mv "ref_patch_emb") 1e-9 1e-9;
  (* ...and adding the positional table is the next line of the model. *)
  gate (Printf.sprintf "V-a +pos_embedding (%s)" name) (Tensor.add pe model.Vit.pos_embedding)
    (mv "ref_pos_added") 1e-9 1e-9;

  (* ---- V-f: the whole model ---- *)
  let our_hidden = Vit.forward_hidden_states model img in
  gate "V-f hidden states" our_hidden ref_hidden 1e-9 1e-9;
  let our_logits = Vit.forward model img in
  gate_vec "V-f logits (CLS head)" our_logits ref_logits 1e-9 1e-9;

  (* ---- V-c: logits are the CLS token, not a pool or the last position ---- *)
  let cls_head = Attention.linear_forward_batch model.Vit.classifier [| our_hidden.(0) |] |> fun r -> r.(0) in
  check "V-c logits come from CLS (position 0)" (max_abs_diff_vec cls_head our_logits < 1e-18);
  let last_head = Attention.linear_forward_batch model.Vit.classifier [| our_hidden.(seq - 1) |] |> fun r -> r.(0) in
  check_diverges_vec "V-c logits are NOT from the last position" ~min_gap:1e-6 our_logits last_head;
  (* ---- V-c2: CLS token is at position 0, and the positional table
         includes a slot for it ---- *)
  check "V-c pos_embedding has a CLS slot" (Array.length model.Vit.pos_embedding = seq);
  (* ---- V-b: patch order is row-major ---- *)
  (* Recompute one patch at a known grid position and require it to be the
     matching sequence row. Swapping row-major for column-major permutes the
     patches, and the positional table is indexed against that order, so a
     self-consistent model would still agree with itself — only this pins
     it. *)
  let npw = img_w / patch in
  let one_patch r col =
    let flat =
      Vit.patch_flat img ~num_channels:chans ~patch_size:patch ~top:(r * patch) ~left:(col * patch)
    in
    Array.init emb (fun o ->
        let w = model.Vit.patch.conv_weight.(o) in
        let s = ref 0.0 in
        for k = 0 to Array.length flat - 1 do
          s := !s +. w.(k) *. flat.(k)
        done;
        !s +. model.Vit.patch.conv_bias.(o))
  in
  let idx r col = 1 + (r * npw) + col in
  (* Probe grid positions that exist in *this* case: the grid is
     (img_h/patch) x (img_w/patch), which is 4x4 for `small` and 2x2 for
     `odd`. Hard-coding indices would only work for one of them. *)
  let nph = img_h / patch in
  let last_r = nph - 1 and last_c = npw - 1 in
  let probes = [ (0, 0); (0, npw - 1); (nph - 1, 0); (last_r, last_c) ] in
  let ours = List.map (fun (r, c) -> one_patch r c) probes in
  let theirs = List.map (fun (r, c) -> pe.(idx r c)) probes in
  check (Printf.sprintf "V-b row-major patch order (%dx%d grid)" nph npw)
    (max_abs_diff_mat (Array.of_list ours) (Array.of_list theirs) = 0.0);
  (* ---- V-d: attention is unmasked ---- *)
  (* The encoder kernel must equal the decoder kernel when the causal mask
     is a no-op — i.e. on a sequence where every position is visible. We
     force that by checking the kernels on a length-1 sequence (trivially
     unmasked) and, more usefully, by asserting the encoder result differs
     from a causal run on the real sequence, since a causal ViT would be a
     different function. *)
  let blk0 = model.Vit.blocks.(0) in
  let x1 = Array.sub our_hidden 0 1 in
  let enc1 = Attention.mha_forward_encoder blk0.Vit.att x1 in
  let dec1 = Attention.mha_forward blk0.Vit.att x1 in
  check "V-d encoder == causal on an all-visible sequence" (max_abs_diff_mat enc1 dec1 < 1e-15);
  let causal_on_full =
    Attention.mha_forward blk0.Vit.att (Array.sub our_hidden 0 4)
  in
  let encoder_on_full = Attention.mha_forward_encoder blk0.Vit.att (Array.sub our_hidden 0 4) in
  check_diverges "V-d encoder differs from causal on a real sequence" ~min_gap:1e-9 encoder_on_full causal_on_full;

  (* ---- V-e: LayerNorm convention is eps OUTSIDE the sqrt ---- *)
  let probe = Safetensors.mat_of_f64 f (g "ref_hidden") in
  let row0 = [| Array.copy probe.(0) |] in
  let ln_outside = Norm.layer_norm model.Vit.final_ln row0 in
  let ln_inside =
    Norm.norm_forward ~scale:model.Vit.final_ln.Norm.scale ~shift:(Some model.Vit.final_ln.Norm.shift)
      ~eps:1e-5 ~centering:Norm.Centering ~eps_placement:Norm.Eps_inside_sqrt row0
  in
  (* The final LN is the last op before ref_hidden, so applying the
     outside-the-sqrt form to ref_hidden again is not the comparison we
     want. Instead: the two conventions must disagree on this input, which
     is what makes the choice observable at all. *)
  check_diverges "V-e eps placement is observable" ~min_gap:1e-9 ln_outside ln_inside;
  (* The library's own default must be the ViT convention. This replaces a
     check that compared the *test's* record against a hardcoded constant,
     which is a gate that cannot fail by construction: flipping every
     eps_placement in this file to Eps_inside_sqrt still passed it, while
     V-f correctly breached at 1.2e-05. Reading the library instead of
     the local binding is what makes it a real assertion. *)
  check "V-e layernorm_make defaults to eps-outside (the ViT convention)"
    ((Norm.layernorm_make ~eps:1e-5 emb).Norm.eps_placement = Norm.Eps_outside_sqrt)


let () =
  if Array.length Sys.argv <> 2 then (
    Printf.eprintf "usage: %s <vit_ref.safetensors>\n" Sys.argv.(0);
    exit 2);
  let f = Safetensors.open_file Sys.argv.(1) in
  Printf.printf "ViT gates (multimodal Part 1)\n";
  (* The GELU the ViT FFN uses, against torch's exact erf. This was the
     one op in the port that was an approximation rather than a formula,
     and while it was one it is what set V-f's tolerance at 1e-6 instead
     of 1e-9 — hence its own gate rather than a silent tolerance.

     It is no longer an approximation: [Norm.erf] is a Maclaurin series
     below 2 and an erfc continued fraction above, measuring 1.11e-15 on
     erf over [-12, 115]. So the bound moves with it, to 1e-13 — ~2e6
     tighter than the Abramowitz & Stegun 7.1.26 polynomial's 2.12e-07,
     which is the point. At the old 2.2e-7 bound this gate would have
     measured 2.2e-16 and passed on a library swapped straight back to the
     polynomial it replaced, which is the "gate that cannot fail" shape
     this repo has now found and fixed three times. 1e-13 is far enough
     above the measured 2.22e-16 to survive a ULP of drift, and far enough
     below 2.12e-07 to mean something. *)
  let gx = Safetensors.vec_of_f64 f "v1.gelu.x" in
  let gref = Safetensors.vec_of_f64 f "v1.gelu.ref" in
  let gours = Array.map Norm.gelu_scalar gx in
  let gerr = ref 0.0 in
  for i = 0 to Array.length gx - 1 do
    gerr := max !gerr (abs_float (gours.(i) -. gref.(i)))
  done;
  Printf.printf "GELU vs torch.erf: max_abs_diff=%.3e (bound 1e-13; A&S 7.1.26 would read 2.12e-07)\n" !gerr;
  if !gerr < 1e-13 then (
    incr passed;
    Printf.printf "PASS V-g GELU is fp64-accurate, not merely bounded\n")
  else (
    incr failed;
    Printf.printf "FAIL V-g GELU exceeds 1e-13: %.3e\n" !gerr);

  run_case f "small" ~img_h:8 ~img_w:8 ~patch:2 ~chans:3 ~emb:32 ~layers:3 ~heads:2 ~classes:5;
  run_case f "odd" ~img_h:6 ~img_w:6 ~patch:3 ~chans:3 ~emb:48 ~layers:2 ~heads:4 ~classes:7;
  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)
