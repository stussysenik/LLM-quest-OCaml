(** Pre-LN transformer block + FFN, mirroring
    [llm_quest/gpt/gpt_transformer_block.py].

    Block (pre-LN, as in the OpenAI implementation despite the
    post-LN figure in the paper):
    {[
      h = x + attn(ln1(x))
      y = h + ffn(ln2(h))
    ]}
    Dropout is identity at inference (all shipped configs use 0.0). *)

open Tensor

type ffn = { l1 : Attention.linear; l2 : Attention.linear; activation : Norm.activation }

let ffn_make (rs : Random.State.t) emb_dim ?(activation = Norm.Gelu_exact) () : ffn =
  {
    l1 = Attention.linear_make rs emb_dim (4 * emb_dim) ~bias:true;
    l2 = Attention.linear_make rs (4 * emb_dim) emb_dim ~bias:true;
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

let block_make (rs : Random.State.t) (cfg : Config.gpt_config) ?(activation = Norm.Gelu_exact)
    ?(out_proj_bias = true) () : block =
  {
    att =
      Attention.mha_make rs ~d_in:cfg.emb_dim ~d_out:cfg.emb_dim ~num_heads:cfg.n_heads
        ~ctx_len:cfg.context_length ~qkv_bias:cfg.qkv_bias ~out_proj_bias ();
    ln1 = Norm.layernorm_make cfg.emb_dim;
    ln2 = Norm.layernorm_make cfg.emb_dim;
    ffn = ffn_make rs cfg.emb_dim ~activation ();
  }

let add_mat (a : mat) (b : mat) : mat =
  Array.mapi (fun i row -> Array.mapi (fun j v -> v +. b.(i).(j)) row) a

let block_forward ?(attn_mask : bool array option = None) (blk : block) (x : mat) : mat =
  let h1 = Norm.layernorm_forward blk.ln1 x in
  let a = Attention.mha_forward ~attn_mask blk.att h1 in
  let h = add_mat x a in
  let h2 = Norm.layernorm_forward blk.ln2 h in
  let f = ffn_forward blk.ffn h2 in
  add_mat h f
