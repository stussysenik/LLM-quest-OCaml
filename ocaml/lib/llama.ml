(** Llama model weights, as loaded from an HF checkpoint (ROADMAP step C).

    This module owns the model's *shape*; the kernels own its
    arithmetic. There is no forward pass yet — RMSNorm, SwiGLU, RoPE and
    GQA land in steps D1-D5, each gated in isolation before anything is
    composed. Loading is already fully shape-checked (see
    [Llama_loader]), so by the time a forward pass exists every tensor
    here is known to have the shape its name claims.

    Weights are stored **as HF writes them** — [nn.Linear] is
    [(out, in)], so this loader never transposes. That is the opposite
    of GPT-2's [Conv1D], and the project convention is [(out, in)]
    precisely because it is the one Llama and every modern transformer
    already use on disk: [Gpt2_loader] converts its [Conv1D] weights at
    load, so both loaders hand the kernels matrices in the same layout
    and [Tensor.linear_fwd] has no layout parameter.

    The one matrix whose layout is *not* the projection convention is
    [embed_tokens] / a tied [lm_head] — [(vocab, hidden)], which is the
    embedding table and, for a tied model, also the [out, in] head
    weight. That is why tying needs no transpose (gate C4a pins it:
    the tied head is the embedding matrix itself, physically), and why
    the convention stays [(out, in)] rather than being flipped to suit
    the embedding. *)

type layer = {
  q_proj : Tensor.mat; (** [heads * head_dim, hidden] *)
  k_proj : Tensor.mat; (** [kv_heads * head_dim, hidden] *)
  v_proj : Tensor.mat; (** [kv_heads * head_dim, hidden] *)
  o_proj : Tensor.mat; (** [hidden, heads * head_dim] *)
  gate_proj : Tensor.mat; (** [intermediate, hidden] *)
  up_proj : Tensor.mat; (** [intermediate, hidden] *)
  down_proj : Tensor.mat; (** [hidden, intermediate] *)
  input_norm : Tensor.vec; (** [input_layernorm], RMSNorm: weight only *)
  post_attn_norm : Tensor.vec; (** [post_attention_layernorm] *)
}

type model = {
  embed_tokens : Tensor.mat; (** [vocab, hidden] — no `wpe` counterpart *)
  layers : layer list;
  norm : Tensor.vec; (** final RMSNorm, weight only *)
  lm_head : Tensor.mat; (** [vocab, hidden] *)
  cfg : Config.t;
}

let layer_count (m : model) : int = List.length m.layers

let mat_count (a : Tensor.mat) : int =
  Array.length a * (if Array.length a = 0 then 0 else Array.length a.(0))

(** C2: the gate compares this against the element count of the tensors
    the loader claimed from the file, which catches a dropped tensor, a
    double-loaded one, or a head that was aliased and then counted as a
    second copy. Note the tied case: aliasing does not add elements, and
    this model does not know whether it was tied — that is what makes
    the comparison meaningful. *)
let num_parameters (m : model) : int =
  let layer =
    List.fold_left
      (fun acc (l : layer) ->
        acc
        + mat_count l.q_proj
        + mat_count l.k_proj
        + mat_count l.v_proj
        + mat_count l.o_proj
        + mat_count l.gate_proj
        + mat_count l.up_proj
        + mat_count l.down_proj
        + Array.length l.input_norm
        + Array.length l.post_attn_norm)
      0 m.layers
  in
  mat_count m.embed_tokens + layer + Array.length m.norm + mat_count m.lm_head