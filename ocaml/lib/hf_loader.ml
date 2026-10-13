(** Loads HuggingFace `gpt2` weights (safetensors) into [Gpt.t], plus the
    reference activations written by [ocaml/parity/export_reference.py].

    Naming map (HF `GPT2LMHeadModel` → ours):
    - [transformer.wte.weight] (V×E) → token embeddings (as-is)
    - [transformer.wpe.weight] (1024×E) → position embeddings (as-is)
    - [transformer.h.{i}.attn.c_attn.{weight,bias}] → Q/K/V thirds.
      HF `Conv1D` stores weights **(in, out)**, so no transpose needed.
    - [transformer.h.{i}.attn.c_proj.{weight,bias}] → output projection
    - [transformer.h.{i}.mlp.c_fc / c_proj] → FFN up/down projections
    - [transformer.ln_f.{weight,bias}] → final LayerNorm
    - [lm_head.weight] (V×E) → **transposed** (`nn.Linear` is (out, in))

    Activation: HF `gpt2` uses `gelu_new` (tanh approx), so the loaded
    model is built with [Norm.Gelu_new] (see [ocaml/ROADMAP.md]). *)

let split_thirds_mat (m : Tensor.mat) : Tensor.mat * Tensor.mat * Tensor.mat =
  let rows = Array.length m in
  let cols = if rows = 0 then 0 else Array.length m.(0) in
  if cols mod 3 <> 0 then failwith "Hf_loader: c_attn width not divisible by 3";
  let w = cols / 3 in
  let part k = Array.init rows (fun i -> Array.init w (fun j -> m.(i).((k * w) + j))) in
  (part 0, part 1, part 2)

let split_thirds_vec (v : Tensor.vec) : Tensor.vec * Tensor.vec * Tensor.vec =
  let n = Array.length v in
  if n mod 3 <> 0 then failwith "Hf_loader: c_attn bias not divisible by 3";
  let w = n / 3 in
  let part k = Array.init w (fun j -> v.((k * w) + j)) in
  (part 0, part 1, part 2)

let layernorm_of (f : Safetensors.file) (prefix : string) ~(eps : float) : Norm.layernorm =
  {
    Norm.scale = Safetensors.vec_of f (prefix ^ ".weight");
    shift = Safetensors.vec_of f (prefix ^ ".bias");
    eps;
    eps_inside_sqrt = true (* HF/PyTorch convention: eps inside the sqrt *);
  }

let load_gpt2 (path : string) : Gpt.t =
  let f = Safetensors.open_file path in
  let cfg = Config.gpt_config_124m in
  let e = cfg.Config.hidden_size in
  let n_heads = cfg.Config.num_heads in
  let tok_emb = Safetensors.mat_of f "transformer.wte.weight" in
  let pos_emb = Safetensors.mat_of f "transformer.wpe.weight" in
  let blocks =
    List.init cfg.Config.num_layers (fun i ->
        let p = Printf.sprintf "transformer.h.%d" i in
        let ln1 = layernorm_of f (p ^ ".ln_1") ~eps:(Config.norm_eps cfg) in
        let ln2 = layernorm_of f (p ^ ".ln_2") ~eps:(Config.norm_eps cfg) in
        let cq_w = Safetensors.mat_of f (p ^ ".attn.c_attn.weight") in
        let cq_b = Safetensors.vec_of f (p ^ ".attn.c_attn.bias") in
        let wq, wk, wv = split_thirds_mat cq_w in
        let bq, bk, bv = split_thirds_vec cq_b in
        let proj w b =
          { Attention.w = w; b = Some b; d_in = e; d_out = e }
        in
        let att =
          {
            Attention.w_q = proj wq bq;
            w_k = proj wk bk;
            w_v = proj wv bv;
            out_proj =
              {
                Attention.w = Safetensors.mat_of f (p ^ ".attn.c_proj.weight");
                b = Some (Safetensors.vec_of f (p ^ ".attn.c_proj.bias"));
                d_in = e;
                d_out = e;
              };
            num_heads = n_heads;
            head_dim = Config.head_dim cfg;
            d_out = e;
            ctx_len = Config.max_positions cfg;
          }
        in
        let ffn =
          {
            Block.l1 =
              {
                Attention.w = Safetensors.mat_of f (p ^ ".mlp.c_fc.weight");
                b = Some (Safetensors.vec_of f (p ^ ".mlp.c_fc.bias"));
                d_in = e;
                d_out = Config.ffn_inner cfg;
              };
            l2 =
              {
                Attention.w = Safetensors.mat_of f (p ^ ".mlp.c_proj.weight");
                b = Some (Safetensors.vec_of f (p ^ ".mlp.c_proj.bias"));
                d_in = Config.ffn_inner cfg;
                d_out = e;
              };
            (* From the config, not hardcoded: an HF `gpt2` checkpoint is
               `gelu_new` (see ROADMAP.md), the repo's own configs are
               exact GELU. *)
            activation = Norm.activation_of cfg;
          }
        in
        { Block.att; ln1; ln2; ffn })
  in
  let final_ln = layernorm_of f "transformer.ln_f" ~eps:(Config.norm_eps cfg) in
  let lm_head =
    {
      Attention.w = Safetensors.mat_of_transposed f "lm_head.weight";
      b = None;
      d_in = e;
      d_out = cfg.Config.vocab_size;
    }
  in
  { Gpt.tok_emb; pos_emb; blocks; final_ln; lm_head; cfg }

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
