(** Loads HuggingFace Llama weights (safetensors) into [Llama.model].

    Naming map (HF `LlamaForCausalLM` → ours):

    | key | shape | notes |
    | :--- | :--- | :--- |
    | [model.embed_tokens.weight] | [vocab, hidden] | no `wpe` counterpart |
    | [model.layers.N.input_layernorm.weight] | [hidden] | RMSNorm, no bias |
    | [model.layers.N.self_attn.q_proj.weight] | [heads*head_dim, hidden] | no transpose |
    | [model.layers.N.self_attn.k_proj.weight] | [kv_heads*head_dim, hidden] | |
    | [model.layers.N.self_attn.v_proj.weight] | [kv_heads*head_dim, hidden] | |
    | [model.layers.N.self_attn.o_proj.weight] | [hidden, heads*head_dim] | |
    | [model.layers.N.post_attention_layernorm.weight] | [hidden] | RMSNorm, no bias |
    | [model.layers.N.mlp.gate_proj.weight] | [intermediate, hidden] | |
    | [model.layers.N.mlp.up_proj.weight] | [intermediate, hidden] | |
    | [model.layers.N.mlp.down_proj.weight] | [hidden, intermediate] | |
    | [model.norm.weight] | [hidden] | final RMSNorm |
    | [lm_head.weight] | [vocab, hidden] | only when untied |

    **The transpose rule: there isn't one.** HF `nn.Linear` stores
    [(out, in)], which is the project-wide convention ([Tensor.linear_fwd]),
    so every key in the table above is read as-is and every matrix in
    [Llama.model] is [(out, in)] — the same layout [Gpt2_loader]
    converts its `Conv1D` weights *into*. That leaves the highest-risk
    line in the port, because the dangerous case is silent: [q_proj],
    [k_proj], [v_proj] and [o_proj] are all square whenever
    [hidden = heads * head_dim], so a transposed one loads without an
    error and produces wrong numbers with no symptom but parity.

    Two defences, neither of which is vigilance. Every tensor is fetched
    through [Safetensors.get_tensor] with a shape derived from
    [Config] — [head_dim] comes from the config, never inferred from
    the file — so transposing any weight here raises on shape the
    moment [hidden <> heads * head_dim]. And where it *is* square, gate
    [layout_llama_out_in] re-derives each matrix's expected [out] from
    the config, independently of the [~expected_shape] above, and
    compares it to the loaded rows: a loader that transposed wrongly in
    two agreeing places still disagrees with the config. *)

let head_key = "lm_head.weight"

let layer_key (i : int) (suffix : string) = Printf.sprintf "model.layers.%d.%s" i suffix

(** Every tensor this loader claims, derived from the config alone.

    Single source of truth for three consumers: [load]'s key-set
    assertion, gate C1's shape checks, and gate C2's parameter count.
    Duplicating the table in the gate would let the two drift, and a
    loader that silently stops claiming a tensor would then look
    correct. *)
let expected_keys (cfg : Config.t) : string list =
  let head = if Config.tie_word_embeddings cfg then [] else [ head_key ] in
  let per_layer i =
    [ layer_key i "input_layernorm.weight"; layer_key i "post_attention_layernorm.weight";
      layer_key i "self_attn.q_proj.weight"; layer_key i "self_attn.k_proj.weight";
      layer_key i "self_attn.v_proj.weight"; layer_key i "self_attn.o_proj.weight";
      layer_key i "mlp.gate_proj.weight"; layer_key i "mlp.up_proj.weight";
      layer_key i "mlp.down_proj.weight" ]
  in
  [ "model.embed_tokens.weight"; "model.norm.weight" ]
  @ head
  @ List.concat_map per_layer (List.init cfg.Config.num_layers (fun i -> i))

(** Llama has no biases anywhere. Asserted rather than assumed: a
    loader that simply stopped asking for [*.bias] would read a Llama
    file just as happily as a GPT-2 one, and would leave every
    projection un-biased by luck instead of by check. *)
let require_no_biases (f : Safetensors.file) (cfg : Config.t) : unit =
  let projections =
    [ "self_attn.q_proj"; "self_attn.k_proj"; "self_attn.v_proj"; "self_attn.o_proj";
      "mlp.gate_proj"; "mlp.up_proj"; "mlp.down_proj" ]
  in
  let per_layer i =
    List.map (fun p -> layer_key i (p ^ ".bias")) projections
    @ [ layer_key i "input_layernorm.bias"; layer_key i "post_attention_layernorm.bias" ]
  in
  let keys =
    List.concat_map per_layer (List.init cfg.Config.num_layers (fun i -> i))
    @ [ "model.norm.bias"; "model.embed_tokens.bias"; head_key ^ ".bias" ]
  in
  List.iter (Safetensors.require_absent f) keys

let load (f : Safetensors.file) (cfg : Config.t) : Llama.model =
  let e = cfg.Config.hidden_size in
  let hd = Config.head_dim cfg in
  let q_dim = cfg.Config.num_heads * hd in
  let kv_dim = Config.kv_heads cfg * hd in
  let inner = Config.ffn_inner cfg in
  if q_dim <> e then
    failwith
      (Printf.sprintf "Llama_loader: num_heads * head_dim = %d <> hidden_size %d" q_dim e);
  if cfg.Config.num_heads mod Config.kv_heads cfg <> 0 then
    failwith
      (Printf.sprintf "Llama_loader: num_heads %d not a multiple of num_key_value_heads %d"
         cfg.Config.num_heads (Config.kv_heads cfg));

  let has_head = List.mem head_key (Safetensors.names f) in
  let tied = Config.tie_word_embeddings cfg in
  (* Tie handling, all four cases explicit: a tie=True file that still
     carries a head is ambiguous (which one wins?), and an untied model
     with no head would have to have one fabricated — the failure mode
     that loads clean and then disagrees numerically for reasons that
     look like arithmetic. *)
  if tied && has_head then
    failwith
      (Printf.sprintf
         "Llama_loader: config ties word embeddings but the file also has %s — ambiguous" head_key);
  if (not tied) && not has_head then
    failwith
      (Printf.sprintf "Llama_loader: tie_word_embeddings=false but %s is absent from the file" head_key);

  Safetensors.assert_exact_keys f ~skip_prefix:"ref." (expected_keys cfg);
  require_no_biases f cfg;

  let embed_tokens =
    Safetensors.get_tensor f "model.embed_tokens.weight" ~expected_shape:[ cfg.Config.vocab_size; e ]
  in
  let lm_head =
    if tied then
      (* Aliased, not copied, and not transposed: [lm_head] is held as
         [out, in] = [vocab, hidden], which is exactly the embedding
         matrix's own layout, so a tied head *is* [embed_tokens].
         Aliasing keeps the parameter count honest (C2) — copying it
         would double-count vocab x hidden, and transposing it would
         silently disagree with the untied branch. *)
      embed_tokens
    else Safetensors.get_tensor f head_key ~expected_shape:[ cfg.Config.vocab_size; e ]
  in
  let layers =
    List.init cfg.Config.num_layers (fun i ->
        let g suffix expected = Safetensors.get_tensor f (layer_key i suffix) ~expected_shape:expected in
        let v suffix = Safetensors.get_vector f (layer_key i suffix) ~expected_len:e in
        {
          Llama.q_proj = g "self_attn.q_proj.weight" [ q_dim; e ];
          k_proj = g "self_attn.k_proj.weight" [ kv_dim; e ];
          v_proj = g "self_attn.v_proj.weight" [ kv_dim; e ];
          o_proj = g "self_attn.o_proj.weight" [ e; q_dim ];
          gate_proj = g "mlp.gate_proj.weight" [ inner; e ];
          up_proj = g "mlp.up_proj.weight" [ inner; e ];
          down_proj = g "mlp.down_proj.weight" [ e; inner ];
          input_norm = v "input_layernorm.weight";
          post_attn_norm = v "post_attention_layernorm.weight";
        })
  in
  let norm = Safetensors.get_vector f "model.norm.weight" ~expected_len:e in
  { Llama.embed_tokens; layers; norm; lm_head; cfg }