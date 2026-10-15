(** Loads a [ViTModel] checkpoint (safetensors) into [Vit.model].

    Naming map ([llm_quest/multimodal/vision_transformer/vit_model.py]'s
    own module, *not* HuggingFace's `ViTModel` — they have nothing in
    common and the fusion path uses this one):

    | key | shape | notes |
    | :--- | :--- | :--- |
    | [patch_embedding.cls_token] | [1, 1, emb] | the learned CLS vector |
    | [patch_embedding.conv_proj.weight] | [emb, C, P, P] | **4-D** — the only weight outside the [(out, in)] convention |
    | [patch_embedding.conv_proj.bias] | [emb] | Conv2d is biased |
    | [pos_embedding] | [1, n_patch+1, emb] | includes the CLS slot |
    | [transformer_blocks.N.att.w_queries.weight] | [emb, emb] | `nn.Linear`, no transpose |
    | [transformer_blocks.N.att.w_keys.weight] | [emb, emb] | |
    | [transformer_blocks.N.att.w_values.weight] | [emb, emb] | |
    | [transformer_blocks.N.att.out_proj.weight] | [emb, emb] | always biased, even when [qkv_bias] is false |
    | [transformer_blocks.N.ln_1.scale] / [.shift] | [emb] | the repo's own LayerNorm, eps **outside** the sqrt |
    | [transformer_blocks.N.ffn.layers.0.weight] | [4*emb, emb] | `nn.Sequential`, so the indices are 0 and 2 |
    | [transformer_blocks.N.ffn.layers.2.weight] | [emb, 4*emb] | |
    | [final_ln.scale] / [.shift] | [emb] | |
    | [classifier.weight] | [num_classes, emb] | |

    **The transpose rule: there isn't one, and that is the risk.** Every
    linear here is an [nn.Linear] storing [(out, in)], the project-wide
    convention ([Tensor.linear_fwd]), so every key is read as-is. For
    [VIT_BASE_CONFIG] all four attention projections are 768x768 and the
    FFN's two are 3072x768 — the [out_proj] one is the square trap, and
    transposing it would load without error and produce wrong numbers with
    no symptom but parity. The defences are the same two as
    [Llama_loader]'s and for the same reason: every tensor is fetched with
    a shape derived from [Vit.t], and [Vit_loader.check_layout] re-derives
    each matrix's expected [out] from the config independently.

    The one weight that *is* transposed at load is the conv kernel, and
    only because it is read as raw row-major [emb * C * P * P] rather than
    as a matrix — see [Safetensors.get_flat] and [Vit.patch_conv]. *)

let layer_key (i : int) (suffix : string) = Printf.sprintf "transformer_blocks.%d.%s" i suffix

(** Every tensor this loader claims, derived from the config alone.

    Single source of truth for three consumers, as in [Llama_loader]:
    [load]'s key-set assertion, the shape checks, and the parameter count
    in the test. Duplicating the table in the test would let the two drift
    and a loader that quietly stopped claiming a tensor would then look
    correct.

    [~prefix] exists because the fusion checkpoint stores the vision half
    under [vit.] alongside a [gpt.] namespace — one file, two models — so
    "the key for the first block's query projection" is a function of
    which half is being loaded, not a constant. *)
let expected_keys ?(prefix = "") (c : Vit.t) : string list =
  let qkv_bias_names =
    (* [w_queries/w_keys/w_values] are the only linears whose bias is
       configurable. When [qkv_bias] is false the file genuinely has no
       [*.bias] for them, so claiming one here would make a correct
       checkpoint fail its own key-set assertion — and *not* claiming one
       would let a [qkv_bias = true] file load with a silently dropped
       bias. Asserting the right one for the config is the whole point. *)
    if c.Vit.qkv_bias then [ "att.w_queries"; "att.w_keys"; "att.w_values" ] else []
  in
  let always_biased = [ "att.out_proj"; "ffn.layers.0"; "ffn.layers.2" ] in
  let per_layer i =
    [ layer_key i "ln_1.scale"; layer_key i "ln_1.shift"; layer_key i "ln_2.scale";
      layer_key i "ln_2.shift" ]
    @ List.concat_map
        (fun n -> [ layer_key i (n ^ ".weight") ])
        ([ "att.w_queries"; "att.w_keys"; "att.w_values"; "att.out_proj"; "ffn.layers.0"; "ffn.layers.2" ])
    @ List.concat_map (fun n -> [ layer_key i (n ^ ".bias") ]) always_biased
    @ List.concat_map (fun n -> [ layer_key i (n ^ ".bias") ]) qkv_bias_names
  in
  List.map (fun k -> prefix ^ k)
    ([ "patch_embedding.cls_token"; "patch_embedding.conv_proj.weight";
       "patch_embedding.conv_proj.bias"; "pos_embedding"; "final_ln.scale"; "final_ln.shift";
       "classifier.weight"; "classifier.bias" ]
    @ List.concat_map per_layer (List.init c.Vit.n_layers (fun i -> i)))

(** A ViT's LayerNorm is the repo's own ([vit_transformer_block.py]):
    eps 1e-5 added **outside** the sqrt, the opposite of
    [torch.nn.LayerNorm]. Built here rather than per-call so the
    convention is stated once. *)
let ln_of (scale : Tensor.vec) (shift : Tensor.vec) : Norm.layernorm =
  { Norm.scale; shift; eps = 1e-5; eps_placement = Norm.Eps_outside_sqrt }

let linear_of (f : Safetensors.file) (key : string) ~(d_in : int) ~(d_out : int) ~(bias : bool) :
    Attention.linear =
  {
    Attention.w = Safetensors.get_tensor f (key ^ ".weight") ~expected_shape:[ d_out; d_in ];
    b = (if bias then Some (Safetensors.get_vector f (key ^ ".bias") ~expected_len:d_out) else None);
    d_in;
    d_out;
  }

(** Re-derive every matrix's expected [out] axis from the config and
    compare it against the loaded rows.

    Independent of the [~expected_shape] checks in [linear_of], and that
    independence is the point: [~expected_shape] says "the file has the
    shape the config implies", which a loader that transposed *every*
    matrix would still satisfy wherever the shape is square. This asks the
    other question — does the row count match the config's idea of what
    the output is — so a wrong-but-consistent transpose is caught. Same
    defence as [Llama_loader]'s [check_llama_layout]. *)
let check_layout (m : Vit.model) : unit =
  let c = m.Vit.cfg in
  let emb = c.Vit.emb_dim in
  let ffn_hidden = 4 * emb in
  let bad (what : string) (l : Attention.linear) (expected_out : int) =
    if l.Attention.d_out <> expected_out || Array.length l.Attention.w <> expected_out then
      failwith (Printf.sprintf "Vit_loader: %s should produce %d rows, loaded %d" what expected_out
                   (Array.length l.Attention.w))
  in
  Array.iteri
    (fun i (b : Vit.block) ->
      bad (Printf.sprintf "block %d w_queries" i) b.Vit.att.Attention.w_q emb;
      bad (Printf.sprintf "block %d w_keys" i) b.Vit.att.Attention.w_k emb;
      bad (Printf.sprintf "block %d w_values" i) b.Vit.att.Attention.w_v emb;
      bad (Printf.sprintf "block %d out_proj" i) b.Vit.att.Attention.out_proj emb;
      bad (Printf.sprintf "block %d ffn_l1" i) b.Vit.ffn_l1 ffn_hidden;
      bad (Printf.sprintf "block %d ffn_l2" i) b.Vit.ffn_l2 emb)
    m.Vit.blocks;
  bad "classifier" m.Vit.classifier c.Vit.num_classes;
  if Array.length m.Vit.patch.conv_weight <> emb then
    failwith
      (Printf.sprintf "Vit_loader: conv_proj should have %d kernels, loaded %d" emb
         (Array.length m.Vit.patch.conv_weight));
  if Array.length m.Vit.pos_embedding <> Vit.num_patches c + 1 then
    failwith
      (Printf.sprintf "Vit_loader: pos_embedding should cover %d slots, loaded %d"
         (Vit.num_patches c + 1)
         (Array.length m.Vit.pos_embedding))

let load ?(prefix = "") (f : Safetensors.file) (c : Vit.t) : Vit.model =
  let emb = c.Vit.emb_dim in
  let ffn_hidden = 4 * emb in
  let p = c.Vit.patch_size in
  let kernel = c.Vit.num_channels * p * p in
  let k name = prefix ^ name in
  let g i suffix = k (layer_key i suffix) in
  if c.Vit.img_height mod p <> 0 || c.Vit.img_width mod p <> 0 then
    failwith
      (Printf.sprintf "Vit_loader: image %dx%d not divisible by patch_size %d" c.Vit.img_height
         c.Vit.img_width p);
  if emb mod c.Vit.n_heads <> 0 then
    failwith
      (Printf.sprintf "Vit_loader: emb_dim %d not divisible by n_heads %d" emb c.Vit.n_heads);

  Safetensors.assert_exact_keys f ~skip_prefix:"" (expected_keys ~prefix c);

  (* Conv2d's [(emb, C, P, P)], read row-major and split into one vector
     per output channel — the channel-major patch order [Vit.patch_flat]
     reads back out. Keeping it a [vec array] rather than a [(out, in)]
     matrix is what makes a mis-reshape visible instead of silent. *)
  let conv_flat =
    Safetensors.get_flat f (k "patch_embedding.conv_proj.weight")
      ~expected_shape:[ emb; c.Vit.num_channels; p; p ]
  in
  let patch =
    {
      Vit.conv_weight = Array.init emb (fun o -> Array.sub conv_flat (o * kernel) kernel);
      conv_bias = Safetensors.get_vector f (k "patch_embedding.conv_proj.bias") ~expected_len:emb;
      cls_token = Safetensors.get_flat f (k "patch_embedding.cls_token") ~expected_shape:[ 1; 1; emb ];
    }
  in
  let pos_mat =
    Safetensors.get_flat f (k "pos_embedding") ~expected_shape:[ 1; Vit.num_patches c + 1; emb ]
  in
  let blocks =
    Array.init c.Vit.n_layers (fun i ->
        let v suffix = Safetensors.get_vector f (g i suffix) ~expected_len:emb in
        {
          Vit.ln_1 = ln_of (v "ln_1.scale") (v "ln_1.shift");
          ln_2 = ln_of (v "ln_2.scale") (v "ln_2.shift");
          att =
            {
              Attention.w_q = linear_of f (g i "att.w_queries") ~d_in:emb ~d_out:emb ~bias:c.Vit.qkv_bias;
              w_k = linear_of f (g i "att.w_keys") ~d_in:emb ~d_out:emb ~bias:c.Vit.qkv_bias;
              w_v = linear_of f (g i "att.w_values") ~d_in:emb ~d_out:emb ~bias:c.Vit.qkv_bias;
              out_proj = linear_of f (g i "att.out_proj") ~d_in:emb ~d_out:emb ~bias:true;
              num_heads = c.Vit.n_heads;
              head_dim = emb / c.Vit.n_heads;
              d_out = emb;
              ctx_len = Vit.num_patches c + 1;
            };
          ffn_l1 = linear_of f (g i "ffn.layers.0") ~d_in:emb ~d_out:ffn_hidden ~bias:true;
          ffn_l2 = linear_of f (g i "ffn.layers.2") ~d_in:ffn_hidden ~d_out:emb ~bias:true;
        })
  in
  let m =
    {
      Vit.cfg = c;
      patch;
      pos_embedding = Array.init (Vit.num_patches c + 1) (fun r -> Array.sub pos_mat (r * emb) emb);
      blocks;
      final_ln =
        ln_of (Safetensors.get_vector f (k "final_ln.scale") ~expected_len:emb)
          (Safetensors.get_vector f (k "final_ln.shift") ~expected_len:emb);
      classifier = linear_of f (k "classifier") ~d_in:emb ~d_out:c.Vit.num_classes ~bias:true;
    }
  in
  check_layout m;
  m

(** Parameter count, as the model's own [parameters()] would report it.
    Used by the loader test to prove nothing was dropped or double-counted
    on the way in — the failure mode of a loader that reads a tensor twice
    or skips one, both of which otherwise load clean. *)
let param_count (m : Vit.model) : int =
  let lin l = (Array.length l.Attention.w * Array.length l.Attention.w.(0))
             + (match l.Attention.b with Some b -> Array.length b | None -> 0)
  in
  let emb = m.Vit.cfg.Vit.emb_dim in
  let kernel = m.Vit.cfg.Vit.num_channels * m.Vit.cfg.Vit.patch_size * m.Vit.cfg.Vit.patch_size in
  (emb * kernel)
  + emb (* conv bias *)
  + emb (* cls token *)
  + (emb * (Vit.num_patches m.Vit.cfg + 1)) (* pos_embedding *)
  + Array.fold_left (fun acc b -> acc + lin b.Vit.att.Attention.w_q + lin b.Vit.att.Attention.w_k
                            + lin b.Vit.att.Attention.w_v + lin b.Vit.att.Attention.out_proj
                            + lin b.Vit.ffn_l1 + lin b.Vit.ffn_l2 + (4 * emb))
      0 m.Vit.blocks
  + (2 * emb) (* final_ln *)
  + lin m.Vit.classifier