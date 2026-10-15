(** Loads the repo's **own** [GPTModel] checkpoint (safetensors) into
    [Gpt.t].

    This is a *third* naming scheme in the project and the reason it is
    worth being explicit about which is which:

    | loader | reads | produced by |
    | :--- | :--- | :--- |
    | [Gpt2_loader] | [transformer.wte.weight], [transformer.h.N.…] | HuggingFace `gpt2` |
    | [Llama_loader] | [model.embed_tokens.weight], [model.layers.N.…] | HF `LlamaForCausalLM` |
    | **this one** | [emb_dict.weight], [trf_blocks.N.…] | `llm_quest.gpt.GPTModel` |

    The fused VLM checkpoint stores its decoder half in the third scheme,
    because `vlm_training.py` builds a [GPTModel] from this repo and saves
    *its* [state_dict]. Feeding that to [Gpt2_loader] would fail on the
    first key, which is the right failure; the trap would be a loader that
    "worked" by guessing.

    Every linear here is [nn.Linear], already [(out, in)] — the project
    convention — so **there is no transpose in this loader at all**, which
    is exactly the hazard [Llama_loader] documents: with
    [emb_dim = n_heads * head_dim] all four attention projections are
    square, so a wrong read loads clean and is wrong. Two defences, as
    elsewhere: shapes derived from [Config], and [check_layout] re-deriving
    each matrix's expected [out] from the config independently.

    Two conventions this checkpoint fixes, both inherited from
    [gpt_transformer_block.py] and neither derivable from a HF comparison:

    - LayerNorm is [(x - mean)/(std + eps)] — eps **outside** the sqrt,
      like the ViT's and unlike [torch.nn.LayerNorm].
    - The FFN uses exact-erf GELU, not [gelu_new]. *)

let layer_key (i : int) (suffix : string) = Printf.sprintf "trf_blocks.%d.%s" i suffix

(** Every tensor this loader claims, derived from the config alone.

    Includes [trf_blocks.N.att.mask] — a *persistent buffer*, so it is in
    every real checkpoint even though the library rebuilds masks on demand.
    Claiming it is deliberate: a file without it is not this model's file. *)
let expected_keys (cfg : Config.t) : string list =
  let per_layer i =
    [ layer_key i "ln_1.scale"; layer_key i "ln_1.shift"; layer_key i "ln_2.scale";
      layer_key i "ln_2.shift"; layer_key i "att.mask" ]
    @ List.concat_map (fun n -> [ layer_key i (n ^ ".weight") ])
        ([ "att.w_queries"; "att.w_keys"; "att.w_values"; "att.out_proj"; "ffn.layers.0";
           "ffn.layers.2" ])
    (* [out_proj] and the FFN are always biased; q/k/v only when the
       config says so. Claiming a bias the file lacks fails the key-set
       assertion, and *not* claiming one when the file has it fails it
       too — which is the point of deriving this from [Config] rather
       than hardcoding it the way a reader that "knows" GPT is biased
       would. *)
    @ [ layer_key i "att.out_proj.bias"; layer_key i "ffn.layers.0.bias";
        layer_key i "ffn.layers.2.bias" ]
    @ (if Config.qkv_bias cfg then
         List.concat_map (fun n -> [ layer_key i (n ^ ".bias") ])
           [ "att.w_queries"; "att.w_keys"; "att.w_values" ]
       else [])
  in
  (* `out` is `nn.Linear(emb, vocab, bias=False)` in [gpt_model.py] — the
     head has no bias, unlike every other Linear in the model. Claiming
     one would fail a correct checkpoint; skipping one when present would
     silently change the logits. *)
  [ "emb_dict.weight"; "pos_emb_dict.weight"; "final_ln.scale"; "final_ln.shift"; "out.weight" ]
  @ List.concat_map per_layer (List.init cfg.Config.num_layers (fun i -> i))

let ln_of (f : Safetensors.file) (cfg : Config.t) (key : string) : Norm.layernorm =
  let e = cfg.Config.hidden_size in
  {
    Norm.scale = Safetensors.get_vector f (key ^ ".scale") ~expected_len:e;
    shift = Safetensors.get_vector f (key ^ ".shift") ~expected_len:e;
    eps = Config.norm_eps cfg;
    eps_placement = Norm.Eps_outside_sqrt;
  }

let linear_of ?bias (f : Safetensors.file) (key : string) ~(d_in : int) ~(d_out : int) : Attention.linear =
  {
    Attention.w = Safetensors.get_tensor f (key ^ ".weight") ~expected_shape:[ d_out; d_in ];
    b =
      (match bias with
       | Some true -> Some (Safetensors.get_vector f (key ^ ".bias") ~expected_len:d_out)
       | _ -> None);
    d_in;
    d_out;
  }

(** The causal mask this block will build must be the one the checkpoint
    carries.

    Cheap to verify and the only check that catches a checkpoint saved at
    a different [context_length]: the mask is a [context_length x
    context_length] buffer, so a model trained at 1024 and loaded claiming
    256 would otherwise run with an [emb_dim]-fine but length-mismatched
    mask, and every attention score would be wrong with no error.

    **The polarity is inverted relative to the rest of the port, and that
    is the whole reason to check.** [gpt_attention.py] registers

        torch.triu(torch.ones(ctx, ctx), diagonal=1).bool()

    so [1] means *masked* and [0] means *attend* — it is consumed by
    [masked_fill_(mask.bool(), -inf)]. Meanwhile [Attention.mha_forward]'s
    [~attn_mask] is a [bool array] where [true] means *attend* ([Vlm.fuse_masks]
    builds leading [true]s for the vision prefix). A reader that read the
    buffer's booleans straight into an [attn_mask] would build a model
    attending to exactly the positions the checkpoint excludes — a
    bidirectional decoder that trains, generates fluent nonsense, and
    fails no shape check anywhere. So the expectation below is stated
    explicitly as [c > r -> masked] rather than inherited from either
    side. *)
let check_mask (f : Safetensors.file) (ctx : int) (i : int) : unit =
  let key = layer_key i "att.mask" in
  let e = Safetensors.find f key in
  let ctx_actual = match e.Safetensors.shape with [ a; b ] -> if a = b then a else 0 | _ -> 0 in
  if ctx_actual <> ctx then
    failwith
      (Printf.sprintf "Gpt_repo_loader: %s is %dx%d, expected %dx%d" key ctx_actual ctx_actual ctx
         ctx);
  (* BOOL or F32: [safetensors.torch.save_file] preserves the buffer's
     native bool dtype, which is what a converted real checkpoint has,
     but nothing stops a pipeline from having cast it to float on the way.
     Read either; reject anything else by dtype rather than by accident. *)
  let m =
    match (Safetensors.find f key).Safetensors.dtype with
    | Safetensors.Bool -> Safetensors.mat_of_bool f key
    | Safetensors.F32 -> Safetensors.mat_of f key
    | d -> failwith (Printf.sprintf "Gpt_repo_loader: %s has dtype %s, expected BOOL or F32" key
                      (match d with
                       | Safetensors.F64 -> "F64"
                       | Safetensors.I64 -> "I64"
                       | Safetensors.Bool -> "BOOL"
                       | Safetensors.F32 -> "F32"))
  in
  let bad = ref (-1) in
  for r = 0 to ctx - 1 do
    for c = 0 to ctx - 1 do
      (* 1 = MASKED, i.e. strictly above the diagonal. *)
      let want = if c > r then 1.0 else 0.0 in
      if m.(r).(c) <> want && !bad < 0 then bad := r * ctx + c
    done
  done;
  if !bad >= 0 then
    let r = !bad / ctx and c = !bad mod ctx in
    failwith
      (Printf.sprintf
         "Gpt_repo_loader: %s is not the causal mask (1 = masked, 0 = attend; row %d, col %d is %g)"
         key r c m.(r).(c))

let check_layout (m : Gpt.t) : unit =
  let e = m.Gpt.cfg.Config.hidden_size in
  let bad (what : string) (l : Attention.linear) (expected_out : int) =
    if l.Attention.d_out <> expected_out || Array.length l.Attention.w <> expected_out then
      failwith
        (Printf.sprintf "Gpt_repo_loader: %s should produce %d rows, loaded %d" what expected_out
           (Array.length l.Attention.w))
  in
  List.iteri
    (fun i (b : Block.block) ->
      bad (Printf.sprintf "block %d w_queries" i) b.Block.att.Attention.w_q e;
      bad (Printf.sprintf "block %d w_keys" i) b.Block.att.Attention.w_k e;
      bad (Printf.sprintf "block %d w_values" i) b.Block.att.Attention.w_v e;
      bad (Printf.sprintf "block %d out_proj" i) b.Block.att.Attention.out_proj e;
      bad (Printf.sprintf "block %d ffn_l1" i) b.Block.ffn.l1 (Config.ffn_inner m.Gpt.cfg);
      bad (Printf.sprintf "block %d ffn_l2" i) b.Block.ffn.l2 e)
    m.Gpt.blocks;
  bad "out" m.Gpt.lm_head m.Gpt.cfg.Config.vocab_size

let load (f : Safetensors.file) (cfg : Config.t) : Gpt.t =
  let e = cfg.Config.hidden_size in
  let ctx = Config.max_positions cfg in
  if cfg.Config.num_heads * Config.head_dim cfg <> e then
    failwith
      (Printf.sprintf "Gpt_repo_loader: num_heads * head_dim <> hidden_size %d" e);
  Safetensors.assert_exact_keys f ~skip_prefix:"ref." (expected_keys cfg);
  List.iter (check_mask f ctx) (List.init cfg.Config.num_layers (fun i -> i));
  let blocks =
    List.init cfg.Config.num_layers (fun i ->
        let p = Printf.sprintf "trf_blocks.%d" i in
        let inner = Config.ffn_inner cfg in
        {
          Block.ln1 = ln_of f cfg (p ^ ".ln_1");
          ln2 = ln_of f cfg (p ^ ".ln_2");
          att =
            {
              Attention.w_q =
                linear_of ~bias:(Config.qkv_bias cfg) f (p ^ ".att.w_queries") ~d_in:e ~d_out:e;
              w_k = linear_of ~bias:(Config.qkv_bias cfg) f (p ^ ".att.w_keys") ~d_in:e ~d_out:e;
              w_v = linear_of ~bias:(Config.qkv_bias cfg) f (p ^ ".att.w_values") ~d_in:e ~d_out:e;
              out_proj = linear_of ~bias:true f (p ^ ".att.out_proj") ~d_in:e ~d_out:e;
              num_heads = cfg.Config.num_heads;
              head_dim = Config.head_dim cfg;
              d_out = e;
              ctx_len = ctx;
            };
          ffn =
            {
              Block.l1 = linear_of ~bias:true f (p ^ ".ffn.layers.0") ~d_in:e ~d_out:inner;
              l2 = linear_of ~bias:true f (p ^ ".ffn.layers.2") ~d_in:inner ~d_out:e;
              activation = Norm.Gelu_exact;
            };
        })
  in
  let m =
    {
      Gpt.tok_emb =
        Safetensors.get_tensor f "emb_dict.weight" ~expected_shape:[ cfg.Config.vocab_size; e ];
      pos_emb =
        Safetensors.get_tensor f "pos_emb_dict.weight" ~expected_shape:[ ctx; e ];
      blocks;
      final_ln = ln_of f cfg "final_ln";
      lm_head = linear_of ~bias:false f "out" ~d_in:e ~d_out:cfg.Config.vocab_size;
      cfg;
    }
  in
  check_layout m;
  m

(** Parameter count, excluding [att.mask] — a buffer, not a parameter, and
    including it would make the count disagree with the model's own
    [parameters()] and hide a real off-by-one in the loader. *)
let param_count (m : Gpt.t) : int =
  let lin l =
    (Array.length l.Attention.w * Array.length l.Attention.w.(0))
    + (match l.Attention.b with Some b -> Array.length b | None -> 0)
  in
  let e = m.Gpt.cfg.Config.hidden_size in
  (m.Gpt.cfg.Config.vocab_size * e) + (Config.max_positions m.Gpt.cfg * e)
  + List.fold_left
      (fun acc (b : Block.block) ->
        acc + lin b.Block.att.Attention.w_q + lin b.Block.att.Attention.w_k
        + lin b.Block.att.Attention.w_v + lin b.Block.att.Attention.out_proj
        + lin b.Block.ffn.l1 + lin b.Block.ffn.l2 + (4 * e))
      0 m.Gpt.blocks
  + (2 * e) + lin m.Gpt.lm_head