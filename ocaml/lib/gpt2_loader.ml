(** Loads HuggingFace `gpt2` weights (safetensors) into [Gpt.t].

    Every weight this loader produces is **[(out, in)]**, the one
    convention [Tensor.linear_fwd] takes — and it also converts each
    key from the layout HF *stores* it in, which for `gpt2` is not one
    layout but two. So every key below goes through one of two helpers
    that name their own on-disk shape, rather than through a boolean
    the caller sets: a flag is precisely the thing that gets flipped
    without being read.

    Naming map (HF `GPT2LMHeadModel` → ours):
    - [transformer.wte.weight] (V×E) → token embeddings (as-is; the
      lookup table is not a linear and never sees [linear_fwd])
    - [transformer.wpe.weight] (P×E) → position embeddings (as-is)
    - [transformer.h.{i}.attn.c_attn.{weight,bias}] → Q/K/V thirds.
      One `Conv1D` of width 3E: split along its (in, out) columns into
      thirds, then transpose each third.
    - [transformer.h.{i}.attn.c_proj.{weight,bias}] → output projection
    - [transformer.h.{i}.mlp.c_fc / c_proj] → FFN up/down projections
    - [transformer.ln_f.{weight,bias}] → final LayerNorm
    - [lm_head.weight] (V×E) → `nn.Linear`, already (out, in), kept

    Activation: HF `gpt2` uses `gelu_new` (tanh approx), so the loaded
    model is built with the config's activation (see [ocaml/ROADMAP.md]). *)

let split_thirds_mat (m : Tensor.mat) : Tensor.mat * Tensor.mat * Tensor.mat =
  let rows = Array.length m in
  let cols = if rows = 0 then 0 else Array.length m.(0) in
  if cols mod 3 <> 0 then failwith "Gpt2_loader: c_attn width not divisible by 3";
  let w = cols / 3 in
  let part k = Array.init rows (fun i -> Array.init w (fun j -> m.(i).((k * w) + j))) in
  (part 0, part 1, part 2)

let split_thirds_vec (v : Tensor.vec) : Tensor.vec * Tensor.vec * Tensor.vec =
  let n = Array.length v in
  if n mod 3 <> 0 then failwith "Gpt2_loader: c_attn bias not divisible by 3";
  let w = n / 3 in
  let part k = Array.init w (fun j -> v.((k * w) + j)) in
  (part 0, part 1, part 2)

let layernorm_of (f : Safetensors.file) (prefix : string) ~(eps : float) ~(hidden : int) : Norm.layernorm =
  {
    Norm.scale = Safetensors.get_vector f (prefix ^ ".weight") ~expected_len:hidden;
    shift = Safetensors.get_vector f (prefix ^ ".bias") ~expected_len:hidden;
    eps;
    eps_inside_sqrt = true (* HF/PyTorch convention: eps inside the sqrt *);
  }

(** HF `Conv1D` stores [(in, out)]; the project convention is
    [(out, in)]. Transposed here, once. *)
let conv1d_weight (f : Safetensors.file) (key : string) ~(d_in : int) ~(d_out : int) : Tensor.mat =
  Tensor.transpose (Safetensors.get_tensor f key ~expected_shape:[ d_in; d_out ])

(** HF `nn.Linear` stores [(out, in)] — already the project convention,
    so this is a plain read. Only [lm_head] is a `nn.Linear` in a
    `gpt2` checkpoint, which is exactly why the two helpers exist
    instead of one with a flag. *)
let nn_linear_weight (f : Safetensors.file) (key : string) ~(d_in : int) ~(d_out : int) : Tensor.mat =
  Safetensors.get_tensor f key ~expected_shape:[ d_out; d_in ]

(** [c_attn] is one [Conv1D] of width 3E holding the Q, K and V thirds
    side by side. Split along its (in, out) columns *first*, then
    transpose each third: transposing the whole tensor and re-splitting
    rows gives the same three matrices, but this way each third is
    transposed while it is still a single small block.

    Kept next to [conv1d_weight] so that every `Conv1D` in this file
    is converted to the project convention in exactly one place. *)
let conv1d_qkv_weight (f : Safetensors.file) (key : string) ~(d_in : int) ~(d_out : int) :
    Tensor.mat * Tensor.mat * Tensor.mat =
  let a, b, c =
    split_thirds_mat (Safetensors.get_tensor f key ~expected_shape:[ d_in; d_out ])
  in
  (Tensor.transpose a, Tensor.transpose b, Tensor.transpose c)

let load (f : Safetensors.file) (cfg : Config.t) : Gpt.t =
  let e = cfg.Config.hidden_size in
  let n_heads = cfg.Config.num_heads in
  let tok_emb =
    Safetensors.get_tensor f "transformer.wte.weight" ~expected_shape:[ cfg.Config.vocab_size; e ]
  in
  let pos_emb =
    Safetensors.get_tensor f "transformer.wpe.weight"
      ~expected_shape:[ Config.max_positions cfg; e ]
  in
  let blocks =
    List.init cfg.Config.num_layers (fun i ->
        let p = Printf.sprintf "transformer.h.%d" i in
        let ln1 = layernorm_of f (p ^ ".ln_1") ~eps:(Config.norm_eps cfg) ~hidden:e in
        let ln2 = layernorm_of f (p ^ ".ln_2") ~eps:(Config.norm_eps cfg) ~hidden:e in
        let inner = Config.ffn_inner cfg in
        let wq, wk, wv = conv1d_qkv_weight f (p ^ ".attn.c_attn.weight") ~d_in:e ~d_out:(3 * e) in
        let cq_b = Safetensors.get_vector f (p ^ ".attn.c_attn.bias") ~expected_len:(3 * e) in
        let bq, bk, bv = split_thirds_vec cq_b in
        let att =
          {
            Attention.w_q = { Attention.w = wq; b = Some bq; d_in = e; d_out = e };
            w_k = { Attention.w = wk; b = Some bk; d_in = e; d_out = e };
            w_v = { Attention.w = wv; b = Some bv; d_in = e; d_out = e };
            out_proj =
              {
                Attention.w = conv1d_weight f (p ^ ".attn.c_proj.weight") ~d_in:e ~d_out:e;
                b = Some (Safetensors.get_vector f (p ^ ".attn.c_proj.bias") ~expected_len:e);
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
                Attention.w = conv1d_weight f (p ^ ".mlp.c_fc.weight") ~d_in:e ~d_out:inner;
                b = Some (Safetensors.get_vector f (p ^ ".mlp.c_fc.bias") ~expected_len:inner);
                d_in = e;
                d_out = inner;
              };
            l2 =
              {
                Attention.w = conv1d_weight f (p ^ ".mlp.c_proj.weight") ~d_in:inner ~d_out:e;
                b = Some (Safetensors.get_vector f (p ^ ".mlp.c_proj.bias") ~expected_len:e);
                d_in = inner;
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
  let final_ln = layernorm_of f "transformer.ln_f" ~eps:(Config.norm_eps cfg) ~hidden:e in
  let lm_head =
    {
      Attention.w =
        nn_linear_weight f "lm_head.weight" ~d_in:e ~d_out:cfg.Config.vocab_size;
      b = None;
      d_in = e;
      d_out = cfg.Config.vocab_size;
    }
  in
  { Gpt.tok_emb; pos_emb; blocks; final_ln; lm_head; cfg }
