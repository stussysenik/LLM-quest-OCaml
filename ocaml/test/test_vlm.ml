(** VLM gates V2-a..V2-g (multimodal Part 2).

     Usage: [./test_vlm <vlm_ref.safetensors>]
     (written by [ocaml/parity/export_vlm.py], which instantiates the
     repo's own [ViTModel] + [ViTAdapter] + [GPTModel] with deterministic
     weights).

     Part 2 is two ideas and each gets its own gate, because each is a
     place where a plausible implementation is a *different function*:

     | V2-a | the adapter is a real projection (32 -> 48, not a pass-through) |
     | V2-b | early fusion concatenates **vision first**, then text |
     | V2-c | text embeddings are token + positional, before fusion |
     | V2-d | the mask is [num_vision ones] ++ [text mask], and it is applied |
     | V2-e | the whole stack composes to the reference logits |
     | V2-f | the last image position predicts the first text token |
     | V2-g | the ViT classifier is **not** in the VLM path |

     The two exported cases differ only in adapter type ("simple" and
     "ffn"), which is the branch [ViTAdapter.__init__] takes; both are
     exercised by the repo's own scripts.

     The ViT and the LLM have different widths (32 vs 48) in the export,
     so V2-a cannot be satisfied by an identity adapter — that is the
     reason for the mismatch, not an accident of it.

     Tolerances: every gate here runs at 1e-9. It used to be 1e-6 for
     V2-a/b/e, because everything they measure had already passed through
     the ViT, whose FFN uses [Norm.gelu_scalar] — then the port's one
     approximation, at a measured 2.11e-07, landing ~2e-8 in every
     downstream tensor regardless of adapter. [Norm.erf] replaced it (1.11e-15
     on erf over [-12, 115], gate V-g in test_vit), so the excuse went with
     it and the gates went to the exact-path tolerance they were always
     waiting for. They now measure 3e-16. *)

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
  if rows <> Array.length b then failwith "max_abs_diff_mat: row count mismatch";
  let m = ref 0.0 in
  for i = 0 to rows - 1 do
    if Array.length a.(i) <> Array.length b.(i) then failwith "max_abs_diff_mat: col count mismatch";
    for j = 0 to Array.length a.(i) - 1 do
      m := max !m (abs_float (a.(i).(j) -. b.(i).(j)))
    done
  done;
  !m

let shape_str (m : mat) : string = Printf.sprintf "[%dx%d]" (Array.length m)
    (if Array.length m = 0 then 0 else Array.length m.(0))

(** Shapes differ => the two are not the same function, so they do
    differ — but never as a bare [Failure] that kills the suite and
    hides the remaining gates. Both callers branch on this instead. *)
let same_shape (a : mat) (b : mat) : bool =
  Array.length a = Array.length b
  && Array.for_all2 (fun x y -> Array.length x = Array.length y) a b

let gate name ours refm abs_tol rel_tol =
  if not (same_shape ours refm) then (
    incr failed;
    Printf.printf "%-38s SHAPE MISMATCH ours=%s ref=%s FAIL\n" name (shape_str ours) (shape_str refm))
  else
    let diff = max_abs_diff_mat ours refm in
    let refmag = mat_max_abs refm in
    let rel = diff /. max refmag 1e-30 in
    let ok = diff <= abs_tol && rel <= rel_tol in
    if ok then incr passed else incr failed;
    Printf.printf "%-38s max_abs_diff=%.3e ref_max=%.3e rel=%.3e %s\n" name diff refmag rel
      (if ok then "OK" else "BREACH")

let check_diverges name ~min_gap a b =
  if not (same_shape a b) then (
    (* A shape difference is a difference. *)
    incr passed;
    Printf.printf "PASS %s (shape mismatch: ours=%s theirs=%s)\n" name (shape_str a) (shape_str b))
  else
    let d = max_abs_diff_mat a b in
    let ok = d > min_gap in
    check (Printf.sprintf "%s (max_abs_diff=%.3e, must exceed %.0e)" name d min_gap) ok

let mk_linear (w : mat) (b : vec option) : Attention.linear =
  let d_out = Array.length w in
  let d_in = if d_out = 0 then 0 else Array.length w.(0) in
  { Attention.w; b; d_in; d_out }

(** GPT-2's own LayerNorm — [gpt_transformer_block.LayerNorm] — divides by
    [std + eps], eps *outside* the sqrt, the same convention as ViT. *)
let layernorm (f : Safetensors.file) (name : string) : Norm.layernorm =
  {
    Norm.scale = Safetensors.vec_of_f64 f (name ^ "_scale");
    shift = Safetensors.vec_of_f64 f (name ^ "_shift");
    eps = 1e-5;
    eps_placement = Norm.Eps_outside_sqrt;
  }

(** Build the ViT from the reference file. The classifier is a zero
    placeholder: the multimodal README drops the classification head, and
    the VLM only ever wants [forward_hidden_states]. Gate V2-g asserts
    that it is genuinely never read. *)
let load_vit (f : Safetensors.file) (p : string) ~(emb : int) ~(layers : int) ~(heads : int)
    ~(n_patch : int) : Vit.model =
  let g name = Printf.sprintf "%s.vit.%s" p name in
  let cfg : Vit.t =
    {
      Vit.img_height = 8;
      img_width = 8;
      patch_size = 2;
      num_channels = 3;
      emb_dim = emb;
      n_layers = layers;
      n_heads = heads;
      qkv_bias = true;
      num_classes = 5;
    }
  in
  let seq = n_patch + 1 in
  let blocks =
    Array.init layers (fun i ->
        let b = Printf.sprintf "%s.vit.block%d" p i in
        let vn n = Safetensors.vec_of_f64 f (Printf.sprintf "%s.%s" b n) in
        let vm n = Safetensors.mat_of_f64 f (Printf.sprintf "%s.%s" b n) in
        {
          Vit.ln_1 =
            {
              Norm.scale = vn "ln1_scale";
              shift = vn "ln1_shift";
              eps = 1e-5;
              eps_placement = Norm.Eps_outside_sqrt;
            };
          ln_2 =
            {
              Norm.scale = vn "ln2_scale";
              shift = vn "ln2_shift";
              eps = 1e-5;
              eps_placement = Norm.Eps_outside_sqrt;
            };
          att =
            {
              Attention.w_q = mk_linear (vm "att_wq_weight") (Some (vn "att_wq_bias"));
              w_k = mk_linear (vm "att_wk_weight") (Some (vn "att_wk_bias"));
              w_v = mk_linear (vm "att_wv_weight") (Some (vn "att_wv_bias"));
              out_proj = mk_linear (vm "att_wo_weight") (Some (vn "att_wo_bias"));
              num_heads = heads;
              head_dim = emb / heads;
              d_out = emb;
              ctx_len = seq;
            };
          ffn_l1 = mk_linear (vm "ffn_l1_weight") (Some (vn "ffn_l1_bias"));
          ffn_l2 = mk_linear (vm "ffn_l2_weight") (Some (vn "ffn_l2_bias"));
        })
  in
  {
    Vit.cfg = cfg;
    patch =
      {
        conv_weight = Safetensors.mat_of_f64 f (g "conv_weight");
        conv_bias = Safetensors.vec_of_f64 f (g "conv_bias");
        cls_token = Safetensors.vec_of_f64 f (g "cls_token");
      };
    pos_embedding = Safetensors.mat_of_f64 f (g "pos_embed");
    blocks;
    final_ln = layernorm f (g "final");
    classifier =
      mk_linear (Array.init 5 (fun _ -> Array.make emb 0.0)) (Some (Array.make 5 0.0));
  }

(** Build the GPT-2 trunk from the reference file. [n_inner = None] is
    4x hidden, [activation = `Gelu] is the exact-erf form
    [gpt_transformer_block.GELU] hardcodes. *)
let load_gpt (f : Safetensors.file) (p : string) ~(vocab : int) ~(emb : int) ~(layers : int)
    ~(heads : int) ~(ctx : int) : Gpt.t =
  let cfg : Config.t =
    {
      Config.vocab_size = vocab;
      hidden_size = emb;
      num_layers = layers;
      num_heads = heads;
      arch =
        Gpt2
          {
            n_positions = ctx;
            n_inner = None;
            layer_norm_eps = 1e-5;
            activation = `Gelu;
            qkv_bias = true;
          };
    }
  in
  let blocks =
    List.init layers (fun i ->
        let b = Printf.sprintf "%s.gpt.block%d" p i in
        let vn n = Safetensors.vec_of_f64 f (Printf.sprintf "%s.%s" b n) in
        let vm n = Safetensors.mat_of_f64 f (Printf.sprintf "%s.%s" b n) in
        {
          Block.att =
            {
              Attention.w_q = mk_linear (vm "att_wq_weight") (Some (vn "att_wq_bias"));
              w_k = mk_linear (vm "att_wk_weight") (Some (vn "att_wk_bias"));
              w_v = mk_linear (vm "att_wv_weight") (Some (vn "att_wv_bias"));
              out_proj = mk_linear (vm "att_wo_weight") (Some (vn "att_wo_bias"));
              num_heads = heads;
              head_dim = emb / heads;
              d_out = emb;
              ctx_len = ctx;
            };
          ln1 =
            {
              Norm.scale = vn "ln1_scale";
              shift = vn "ln1_shift";
              eps = 1e-5;
              eps_placement = Norm.Eps_outside_sqrt;
            };
          ln2 =
            {
              Norm.scale = vn "ln2_scale";
              shift = vn "ln2_shift";
              eps = 1e-5;
              eps_placement = Norm.Eps_outside_sqrt;
            };
          ffn =
            {
              Block.l1 = mk_linear (vm "ffn_l1_weight") (Some (vn "ffn_l1_bias"));
              l2 = mk_linear (vm "ffn_l2_weight") (Some (vn "ffn_l2_bias"));
              activation = Norm.Gelu_exact;
            };
        })
  in
  {
    Gpt.tok_emb = Safetensors.mat_of_f64 f (p ^ ".gpt.tok_emb");
    pos_emb = Safetensors.mat_of_f64 f (p ^ ".gpt.pos_emb");
    blocks;
    final_ln = layernorm f (p ^ ".gpt.final");
    lm_head = mk_linear (Safetensors.mat_of_f64 f (p ^ ".gpt.out_weight")) None;
    cfg;
  }

let run_case (f : Safetensors.file) (name : string) ~(adapter_type : string) ~(vit_emb : int)
    ~(llm_emb : int) ~(layers : int) ~(heads : int) ~(n_patch : int) ~(vocab : int) ~(ctx : int)
    ~(text_len : int) =
  let p = Printf.sprintf "v2.%s" name in
  let g n = Printf.sprintf "%s.%s" p n in
  let n_vit = n_patch + 1 in
  let seq = n_vit + text_len in

  let vit = load_vit f p ~emb:vit_emb ~layers ~heads ~n_patch in
  let gpt = load_gpt f p ~vocab ~emb:llm_emb ~layers ~heads ~ctx in
  let adapter : Vlm.adapter =
    if adapter_type = "simple" then
      Vlm.Simple (mk_linear (Safetensors.mat_of_f64 f (g "adapter_l1_weight")) None)
    else
      Vlm.Ffn
        {
          l1 = mk_linear (Safetensors.mat_of_f64 f (g "adapter_l1_weight")) None;
          l2 = mk_linear (Safetensors.mat_of_f64 f (g "adapter_l2_weight")) None;
          activation = Norm.Gelu_exact;
        }
  in
  let model : Vlm.model = { Vlm.vit; adapter; gpt } in

  let img = Safetensors.mat3_of_f64 f (g "img") in
  let ids = Array.map Int64.to_int (Safetensors.i64_array_of f (g "input_ids")) in
  let text_mask = Array.map (fun v -> Int64.to_int v <> 0) (Safetensors.i64_array_of f (g "text_mask")) in
  if Array.length ids <> text_len || Array.length text_mask <> text_len then
    failwith "test_vlm: input length mismatch";

  (* One tolerance for both cases, and it is the exact-path 1e-9. The
     simple adapter is tempting to gate separately from the ffn one — it is
     a bare linear with no GELU — but the ViT feeding it is the same ViT,
     and its FFN goes through [Norm.gelu_scalar]. While that op was an
     approximation at 2.11e-07 it put ~2e-8 in every downstream tensor
     regardless of adapter, so a tighter bound here was not measuring the
     adapter, it was re-gating V-g of test_vit and failing on it. With
     [Norm.erf] the difference is gone and one tolerance is simply the
     right answer for both branches. *)
  let tol = 1e-9 in

  Printf.printf "\n%s: %s adapter, %d vit_emb -> %d llm_emb, %d vision + %d text = %d\n" name
    adapter_type vit_emb llm_emb n_vit text_len seq;

  (* ---- V2-a: the adapter is a real projection ---- *)
  let vit_hidden = Vit.forward_hidden_states vit img in
  gate (Printf.sprintf "V2-a vit hidden states (%s)" name) vit_hidden (Safetensors.mat_of_f64 f (g "ref_vit_hidden")) 1e-9 1e-9;
  let vision = Vlm.vision_embeddings model ~image:img in
  gate (Printf.sprintf "V2-a adapter output (%s)" name) vision (Safetensors.mat_of_f64 f (g "ref_vision")) tol tol;
  (* Widths differ, so a "skipped adapter" cannot typecheck — but a
     *reused* one can: the projection must be the loaded weights, not
     something of the right shape. Rebuilding the adapter with zeros and
     requiring a large divergence is what pins that. *)
  let zeros_w =
    Array.init (Array.length vision.(0)) (fun _ -> Array.make vit_emb 0.0)
  in
  let dead_adapter = Vlm.Simple (mk_linear zeros_w None) in
  check_diverges (Printf.sprintf "V2-a adapter is the loaded projection, not zeros (%s)" name)
    ~min_gap:1e-6 vision (Vlm.adapter_forward dead_adapter vit_hidden);

  (* ---- V2-c: text embeddings are token + position, before fusion ---- *)
  let text_emb = Gpt.embed gpt ids in
  gate (Printf.sprintf "V2-c text embeddings (%s)" name) text_emb (Safetensors.mat_of_f64 f (g "ref_text_emb")) 1e-9 1e-9;

  (* ---- V2-b: early fusion, vision first ---- *)
  let fused = Vlm.fused_embeddings model ~image:img ~ids in
  gate (Printf.sprintf "V2-b fused embeddings (%s)" name) fused (Safetensors.mat_of_f64 f (g "ref_fused")) tol tol;
  check (Printf.sprintf "V2-b fused length is vision + text (%s)" name) (Array.length fused = seq);
  (* Swapping the halves is the one fusion bug that still runs: same
     shape, same dtypes, a perfectly valid attention pattern — and every
     position is wrong. It has to diverge or the positive gate above is
     not testing order at all. *)
  let swapped = Array.append text_emb vision in
  check_diverges (Printf.sprintf "V2-b vision comes first, not last (%s)" name) ~min_gap:1e-6 fused swapped;

  (* ---- V2-d: the mask ---- *)
  let our_mask = Vlm.fuse_masks ~num_vision:n_vit ~text_mask in
  let mask_ok = Array.length our_mask = seq && Array.for_all (fun b -> b) (Array.sub our_mask 0 n_vit) in
  let mask_tail =
    Array.for_all (fun i -> our_mask.(n_vit + i) = text_mask.(i)) (Array.init text_len (fun i -> i))
  in
  check (Printf.sprintf "V2-d mask = %d vision ones ++ text mask (%s)" n_vit name) (mask_ok && mask_tail);
  (* The mask must actually be applied. A port that builds it and then
     drops it produces valid causal attention that simply lets the real
     tokens see padding — so the positive gates above would still pass.
     Only a divergence gate catches it. *)
  let logits_masked = Vlm.forward ~text_mask:(Some text_mask) model ~image:img ~ids in
  let logits_unmasked = Vlm.forward model ~image:img ~ids in
  check_diverges (Printf.sprintf "V2-d the padding mask is applied (%s)" name) ~min_gap:1e-9 logits_masked
    logits_unmasked;

  (* ---- V2-e: the whole stack ---- *)
  gate (Printf.sprintf "V2-e fused logits (%s)" name) logits_masked (Safetensors.mat_of_f64 f (g "ref_logits")) tol tol;
  let hidden = Vlm.forward_hidden_states ~text_mask:(Some text_mask) model ~image:img ~ids in
  check (Printf.sprintf "V2-e hidden states are pre-head (%s)" name)
    (max_abs_diff_mat (Attention.linear_forward_batch gpt.Gpt.lm_head hidden) logits_masked < 1e-18);

  (* ---- V2-f: the last image position predicts the first text token ---- *)
  let shifted = Vlm.shifted_text_logits ~num_vision_tokens:n_vit logits_masked in
  (* [vlm_loss] slices [logits[:, nv-1 : -1]]: rows [nv-1 .. s-2], which is
     [s - nv] = text_len rows. That is text_len, NOT text_len - 1 — the last
     *logit* row is dropped (nothing follows it), not the last *text* row. *)
  check (Printf.sprintf "V2-f shifted rows = text_len (%s)" name) (Array.length shifted = text_len);
  let same_as_first_real =
    max_abs_diff_mat [| shifted.(0) |] [| logits_masked.(n_vit - 1) |] < 1e-18
  in
  check (Printf.sprintf "V2-f first shifted row is the last vision position (%s)" name) same_as_first_real;
  (* The off-by-one this guards: starting the slice at [n_vit] instead of
     [n_vit - 1] drops the last image row and picks up the final logit row —
     same length, wrong content, and only a divergence gate notices. *)
  let off_by_one = Array.sub logits_masked n_vit text_len in
  check_diverges (Printf.sprintf "V2-f slice starts at n_vit-1, not n_vit (%s)" name) ~min_gap:1e-9 shifted
    off_by_one;

  (* ---- V2-g: the ViT classifier is not in the VLM path ---- *)
  (* The multimodal README drops Google's classification head: the VLM
     wants patches, not a class. Perturbing the (unused, zeroed) head and
     requiring bit-identical logits turns "we called forward_hidden_states"
     from a claim about the source into a measurement. *)
  let noisy_cls : Attention.linear =
    let w = Array.init 5 (fun _ -> Array.init vit_emb (fun _ -> 1.0)) in
    { Attention.w; b = Some (Array.make 5 1.0); d_in = vit_emb; d_out = 5 }
  in
  let with_cls : Vlm.model = { model with Vlm.vit = { vit with Vit.classifier = noisy_cls } } in
  let logits_cls = Vlm.forward ~text_mask:(Some text_mask) with_cls ~image:img ~ids in
  check (Printf.sprintf "V2-g logits ignore the ViT classifier (%s)" name)
    (max_abs_diff_mat logits_cls logits_masked < 1e-18)

  (* There is deliberately NO static gate on the adapter activation.

     Such a gate would read `activation` off the record and compare it with
     `Norm.Gelu_exact` — but this very test is what *built* that record, so
     the comparison is against itself and can never fail. Mutation testing
     made it concrete: hardcoding `Gelu_new` inside `adapter_forward` (the
     library discarding the stored variant) left that gate green, and only
     V2-a / V2-b / V2-e caught it, at 2.6e-5 / 1.7e-5 against a 1e-6 gate.

     Same defect as the retired ViT V-e, which compared a record built here
     against a constant. The behaviour is gated where it belongs: the ffn
     case of V2-a measures `ref_vision`, produced by the real `nn.GELU`, so
     a static gate would have nothing left to say. (No apostrophes in this
     comment: a lone quote inside an OCaml comment lexes as a char literal
     and the unterminated one surfaces as a syntax error at EOF.) *)


let () =
  if Array.length Sys.argv <> 2 then (
    Printf.eprintf "usage: %s <vlm_ref.safetensors>\n" Sys.argv.(0);
    exit 2);
  let f = Safetensors.open_file Sys.argv.(1) in
  Printf.printf "VLM gates (multimodal Part 2): ViT adapter + early fusion\n";
  (* `small` is the ViT shape from export_vit (8x8, patch 2 -> 16 patches),
     so the 17-row vision side is shared with Part 1 and only the fusion
     is new. *)
  run_case f "ffn" ~adapter_type:"ffn" ~vit_emb:32 ~llm_emb:48 ~layers:2 ~heads:2 ~n_patch:16
    ~vocab:101 ~ctx:64 ~text_len:6;
  run_case f "simple" ~adapter_type:"simple" ~vit_emb:32 ~llm_emb:48 ~layers:2 ~heads:2 ~n_patch:16
    ~vocab:101 ~ctx:64 ~text_len:6;
  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)
