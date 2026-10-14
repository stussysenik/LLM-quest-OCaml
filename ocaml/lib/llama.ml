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

open Tensor

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

(* ------------------------------------------------------------------ *)
(* Forward pass (ROADMAP step E)                                       *)
(* ------------------------------------------------------------------ *)

(** Token ids → rows of [embed_tokens]. Llama has no positional table:
    position enters through RoPE, not a learned embedding, so the lookup
    is the whole "embedding" stage. The id is range-checked because an out
    of range read here would be a silent out-of-bounds rather than an
    error, and a token id past the vocabulary is exactly the kind of
    input mistake that otherwise shows up 12 layers downstream as a
    numerical breach. *)
let embed (m : model) (ids : int array) : Tensor.mat =
  let vocab = Array.length m.embed_tokens in
  Array.map
    (fun id ->
      if id < 0 || id >= vocab then
        invalid_arg (Printf.sprintf "Llama.embed: token id %d outside [0, %d)" id vocab);
      Array.copy m.embed_tokens.(id))
    ids

(** The RoPE tables this model's config asks for. [rope_scaling = None]
    means classic RoPE; [Llama3] is the step-D4 frequency scaling. Going
    through [Config] rather than taking the factors as arguments is what
    makes the config the single source of truth for all three. *)
let rope_tables (m : model) : Tensor.mat * Tensor.mat =
  let base = Config.rope_theta m.cfg in
  let head_dim = Config.head_dim m.cfg in
  let ctx_len = Config.max_positions m.cfg in
  match Config.rope_scaling m.cfg with
  | None -> Rope.compute_angles ~base ~head_dim ~ctx_len
  | Some scaling -> Rope.compute_angles_llama3 ~base ~head_dim ~ctx_len ~scaling

(** The [Attention.gqa] view of a layer's projections. No biases anywhere:
    Llama's linears are bias-free ([Config.qkv_bias] is [false]), so every
    [b] is [None] rather than a zero vector that would silently make the
    kernels' bias argument look load-bearing. *)
let gqa_of_layer (cfg : Config.t) (l : layer) : Attention.gqa =
  let num_heads = cfg.num_heads in
  let num_kv_groups = Config.kv_heads cfg in
  let head_dim = Config.head_dim cfg in
  let hidden = cfg.hidden_size in
  let kv_width = num_kv_groups * head_dim in
  {
    Attention.w_q = { Attention.w = l.q_proj; b = None; d_in = hidden; d_out = num_heads * head_dim };
    w_k = { Attention.w = l.k_proj; b = None; d_in = hidden; d_out = kv_width };
    w_v = { Attention.w = l.v_proj; b = None; d_in = hidden; d_out = kv_width };
    out_proj = { Attention.w = l.o_proj; b = None; d_in = num_heads * head_dim; d_out = hidden };
    num_heads;
    num_kv_groups;
    head_dim;
    d_out = hidden;
  }

let swiglu_of_layer (l : layer) : Mlp.swiglu =
  { Mlp.gate_proj = l.gate_proj; up_proj = l.up_proj; down_proj = l.down_proj }

(** One decoder block: pre-norm attention, residual, pre-norm SwiGLU,
    residual. Both norms are *pre*-norm, which is why [x] is the residual
    stream being added to rather than something recomputed. *)
let block_forward (cfg : Config.t) (l : layer) (cos : mat) (sin : mat) ~positions (x : mat) : mat =
  let eps = Config.norm_eps cfg in
  let gqa = gqa_of_layer cfg l in
  let normed = Norm.rms_norm l.input_norm ~eps x in
  let attn = Attention.gqa_forward ~positions ~cos ~sin gqa normed in
  let x1 = Tensor.add x attn in
  let normed2 = Norm.rms_norm l.post_attn_norm ~eps x1 in
  let mlp = Mlp.forward (swiglu_of_layer l) normed2 in
  Tensor.add x1 mlp

(** [forward_hidden_states ?positions m ids] returns [(states, logits)].

    [states] has length [n_layers + 1] and uses the *same* convention as
    [Gpt.forward_states], which is what lets the parity harness run one
    stage loop for both families:

    - [states.(0)] — the embedding output
    - [states.(i)] for [1 <= i <= n_layers] — the raw output of block [i-1]

    Note [states.(n_layers)] is the **raw** last-block output, *not* the
    post-final-norm state. The harness applies the family's own
    [final_norm] to it and compares the result against
    [ref.hidden.{n_layers}], which the export identified as post-norm by
    pairing each hidden state against the logits rather than assuming (see
    [export_reference.py]'s [probe_convention]). The two are not
    interchangeable: for GPT-2 the raw state has |x| ~ 3000 against the
    normed state's ~170, so comparing the wrong pair is a 1e-2 breach
    rather than a rounding difference.

    [positions] is [None] for a plain prefill (positions [0..n-1]) and
    [Some ps] when the ids sit at arbitrary absolute positions, which is
    what a cached decode step needs. *)
let forward_hidden_states ?(positions : int array option = None) (m : model) (ids : int array) :
    mat array * mat =
  let cfg = m.cfg in
  let eps = Config.norm_eps cfg in
  let cos, sin = rope_tables m in
  let x0 = embed m ids in
  let n_layers = List.length m.layers in
  let states = Array.make (n_layers + 1) x0 in
  let cur = ref x0 in
  List.iteri
    (fun i l ->
      let next = block_forward cfg l cos sin ~positions !cur in
      states.(i + 1) <- next;
      cur := next)
    m.layers;
  let final_normed = Norm.rms_norm m.norm ~eps !cur in
  let logits = Tensor.linear_fwd m.lm_head final_normed None in
  (states, logits)

(** The forward pass proper: logits for [ids]. *)
let forward ?positions (m : model) (ids : int array) : mat =
  snd (forward_hidden_states ?positions m ids)