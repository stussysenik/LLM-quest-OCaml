(** Pre-LN transformer block + FFN, mirroring
    [llm_quest/gpt/gpt_transformer_block.py].

    Block (pre-LN, as in the OpenAI implementation despite the
    post-LN figure in the paper):
    {[
      h = x + attn(ln1(x))
      y = h + ffn(ln2(h))
    ]}
    Dropout is identity at inference (all shipped configs use 0.0).

    Shapes come from [Config] accessors, not from a family match: the
    feed-forward width is [Config.ffn_inner] (4x hidden for GPT-2,
    [intermediate_size] for Llama) and the context bound is
    [Config.max_positions]. *)

open Tensor

type ffn = { l1 : Attention.linear; l2 : Attention.linear; activation : Norm.activation }

let ffn_make (rs : Random.State.t) d_in d_inner ?(activation = Norm.Gelu_exact) () : ffn =
  {
    l1 = Attention.linear_make rs d_in d_inner ~bias:true;
    l2 = Attention.linear_make rs d_inner d_in ~bias:true;
    activation;
  }

let ffn_forward (f : ffn) (x : mat) : mat =
  let h = Attention.linear_forward_batch f.l1 x in
  let h = Norm.apply_activation_mat f.activation h in
  Attention.linear_forward_batch f.l2 h

type block = {
  att : Attention.mha;
  ln1 : Norm.layernorm;
  ln2 : Norm.layernorm;
  ffn : ffn;
}

(** [activation] defaults to the config's, so the checkpoint family has
    exactly one source of truth for it (an HF `gpt2` is [Gelu_new]; a
    from-scratch tiny config is exact [Gelu]). *)
let block_make (rs : Random.State.t) (cfg : Config.t) ?activation ?(out_proj_bias = true) () : block =
  let activation = match activation with Some a -> a | None -> Norm.activation_of cfg in
  {
    att =
      Attention.mha_make rs ~d_in:cfg.Config.hidden_size ~d_out:cfg.Config.hidden_size
        ~num_heads:cfg.Config.num_heads ~ctx_len:(Config.max_positions cfg)
        ~qkv_bias:(Config.qkv_bias cfg) ~out_proj_bias ();
    ln1 = Norm.layernorm_make ~eps:(Config.norm_eps cfg) cfg.Config.hidden_size;
    ln2 = Norm.layernorm_make ~eps:(Config.norm_eps cfg) cfg.Config.hidden_size;
    ffn = ffn_make rs cfg.Config.hidden_size (Config.ffn_inner cfg) ~activation ();
  }

let add_mat (a : mat) (b : mat) : mat =
  Array.mapi (fun i row -> Array.mapi (fun j v -> v +. b.(i).(j)) row) a

(** Block forward that also returns the CURRENT step's per-head K/V
    (pre-prefix: exactly what the cache must store for this step).
    With [kv=None] the trunk computes identically to [block_forward]
    — same kernels, same order — so the plain forward delegates here
    and parity keeps proving both. *)
let block_forward_kv ?(attn_mask : bool array option = None) ?(kv : Kv_cache.layer_kv option = None)
    (blk : block) (x : mat) : mat * Tensor.mat array * Tensor.mat array =
  let h1 = Norm.layernorm_forward blk.ln1 x in
  let qh, kh, vh = Attention.project_split blk.att h1 in
  let kh_full, vh_full = Attention.with_prefix blk.att kv kh vh in
  let a = Attention.combine ~attn_mask blk.att qh kh_full vh_full in
  let h = add_mat x a in
  let h2 = Norm.layernorm_forward blk.ln2 h in
  let f = ffn_forward blk.ffn h2 in
  (add_mat h f, kh, vh)

let block_forward ?(attn_mask : bool array option = None) (blk : block) (x : mat) : mat =
  let out, _, _ = block_forward_kv ~attn_mask blk x in
  out

(** Batched block: per-item masks, shared verified kernel. *)
let block_forward_batch ?(attn_mask : bool array array option = None) (blk : block) (xs : mat array) :
    mat array =
  Array.mapi
    (fun i x ->
      let m = match attn_mask with None -> None | Some ms -> Some ms.(i) in
      block_forward ~attn_mask:m blk x)
    xs
