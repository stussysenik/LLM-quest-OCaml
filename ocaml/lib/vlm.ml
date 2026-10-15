(** Vision-Language Model: ViT-LLM adapter + early fusion, mirroring
    [llm_quest/multimodal/] — [vit_engine.py]'s [ViTAdapter] and the
    forward in [vlm_engine.py].

    Multimodal Part 2. Part 1 built the image encoder; this is the part
    that makes those embeddings mean something to a language model, and it
    is two ideas:

    1. **The adapter** maps ViT hidden states into the LLM's embedding
       space. It is one linear ("simple") or a two-layer GELU FFN ("ffn",
       the type [vlm_generation.py] and [vlm_training.py] actually use).
       It is *not* a no-op even when both sides are the same width — the
       two spaces were never trained together, so the adapter is where they
       are aligned. Both layers are [bias=False] by default, which is the
       opposite of every other linear in this repo; a port that "helpfully"
       adds them is wrong by a constant and only a loaded-weight gate
       notices.

    2. **Early fusion** concatenates the projected image embeddings with
       the text embeddings along the *sequence* axis and runs the whole
       thing through the decoder with a causal mask. Vision comes first:
       [cat([vision, text], dim=1)]. Nothing about that order is implied
       by the shapes — swapping it still runs, still attends, still
       produces a self-consistent model, and misaligns every position, so
       it is gated rather than assumed.

    The mask concatenates the same way: [num_vision] ones, then the text
    mask. [vlm_engine.py] trains on that exact mask, and
    [vlm_loss] slices [logits[:, num_vision - 1 : -1]] so that the *last
    vision token* predicts the first text token — [shifted_text_logits]
    is that slice, and the offset is the single most likely off-by-one in
    this file.

    Dropout is the identity for the same reason it is in the decoder and
    in the ViT: [nn.Dropout] is identity in eval mode whatever the
    configured rate, and parity is an inference property. *)

open Tensor

(* ------------------------------------------------------------------ *)
(* Adapter                                                             *)
(* ------------------------------------------------------------------ *)

(** Which adapter. [activation] is stored rather than derived, because
    [Norm.activation_of] would give [Gelu_new] (the tanh approximation)
    for a GPT-2-shaped config and the adapter wants [Gelu_exact] —
    [nn.GELU()] with no [approximate] argument is the erf form. A derived
    default is the quiet way to get the wrong 1e-5. *)
type adapter =
  | Simple of Attention.linear
  | Ffn of {
      l1 : Attention.linear;
      l2 : Attention.linear;
      activation : Norm.activation;
    }

let adapter_forward (a : adapter) (x : mat) : mat =
  match a with
  | Simple l -> Attention.linear_forward_batch l x
  | Ffn { l1; l2; activation } ->
      let h = Norm.apply_activation_mat activation (Attention.linear_forward_batch l1 x) in
      Attention.linear_forward_batch l2 h

(* ------------------------------------------------------------------ *)
(* Model                                                               *)
(* ------------------------------------------------------------------ *)

type model = {
  vit : Vit.model;
  adapter : adapter;
  gpt : Gpt.t;
}

(** How many sequence positions the image occupies after fusion: the
    CLS token plus one per patch — [Vit.forward_hidden_states]' row count.
    Counted from the encoder rather than passed in, so it cannot drift
    from the tensor it is about to slice. *)
let num_vision_tokens (m : model) : int = 1 + Vit.num_patches m.vit.Vit.cfg

(* ------------------------------------------------------------------ *)
(* Early fusion                                                        *)
(* ------------------------------------------------------------------ *)

(** [fuse ~vision ~text] — vision first, then text, along the sequence
    axis. Both are [(n, emb)] and must be the same width; the adapter is
    what makes them so. *)
let fuse ~(vision : mat) ~(text : mat) : mat =
  if Array.length vision > 0 && Array.length text > 0 && Array.length vision.(0) <> Array.length text.(0)
  then invalid_arg "Vlm.fuse: vision and text embeddings differ in width";
  Array.append vision text

(** [fuse_masks ~num_vision ~text_mask] — image positions are always
    attendable ([true]), then the text mask verbatim. *)
let fuse_masks ~(num_vision : int) ~(text_mask : bool array) : bool array =
  Array.append (Array.make num_vision true) text_mask

(** The image embeddings in the LLM's own space: ViT hidden states,
    through the adapter, ready to be concatenated with text. *)
let vision_embeddings (m : model) ~(image : Vit.image) : mat =
  adapter_forward m.adapter (Vit.forward_hidden_states m.vit image)

(** [fused_embeddings] — the row the decoder actually sees: image
    embeddings first, then [Gpt.embed ids] (token + positional). This is
    the [input_embedded=True] path of [GPTModel.forward]. *)
let fused_embeddings (m : model) ~(image : Vit.image) ~(ids : int array) : mat =
  fuse ~vision:(vision_embeddings m ~image) ~text:(Gpt.embed m.gpt ids)

(** The decoder trunk over the fused sequence. [text_mask] is the mask of
    the *text* part only; it is fused with [num_vision] leading ones here,
    which is the whole of [vlm_engine.py]'s mask handling. *)
let forward_hidden_states ?(text_mask : bool array option = None) (m : model) ~(image : Vit.image)
    ~(ids : int array) : mat =
  let attn_mask =
    Option.map (fun mk -> fuse_masks ~num_vision:(num_vision_tokens m) ~text_mask:mk) text_mask
  in
  (* [~attn_mask:attn_mask], not the [?attn_mask] pun: [Gpt.trunk_from_emb]'s
     optional parameter is already [bool array option], and the [?label:e]
     form expects the *double* option. The pun is a type error, so it cannot
     pass a [None] through silently — but it is the natural thing to reach
     for, so it is worth naming here. *)
  Gpt.trunk_from_emb ~attn_mask m.gpt (fused_embeddings m ~image ~ids)

(** Logits over the fused sequence: [(num_vision + seq_len, vocab)]. *)
let forward ?(text_mask : bool array option = None) (m : model) ~(image : Vit.image)
    ~(ids : int array) : mat =
  Attention.linear_forward_batch m.gpt.Gpt.lm_head (forward_hidden_states ~text_mask m ~image ~ids)

(** [vlm_loss] slices [logits[:, num_vision_tokens - 1 : -1]] so that the
    last image position predicts the first text token — the image tokens
    are context, never targets. Count is [seq - num_vision] because the
    final logit row has nothing after it to predict. *)
let shifted_text_logits ~(num_vision_tokens : int) (logits : mat) : mat =
  let s = Array.length logits in
  if num_vision_tokens < 1 || num_vision_tokens > s then
    invalid_arg (Printf.sprintf "Vlm.shifted_text_logits: num_vision_tokens=%d, seq=%d" num_vision_tokens s);
  Array.sub logits (num_vision_tokens - 1) (s - num_vision_tokens)

(** [vlm_loss] — next-token prediction over the *text* only.

    Two slice/mask steps, and both are load-bearing:

    1. [shifted_text_logits] drops the first [num_vision_tokens - 1] logit
       rows, so row 0 is the last image position predicting the first
       text token. Image positions are context, never targets.
    2. [text_mask] is applied to the **targets**, not the logits: a padded
       position is marked [ignore_index] rather than predicted as
       whatever id happened to sit there. [Loss.cross_entropy] then drops
       those rows from the mean's denominator as well as its numerator.

    [labels] is one entry per shifted row — the token each row predicts —
    already aligned, which is how `vlm_loss` receives them ("already
    aligned with shifted_logits"). The count is [text_len], not
    [text_len - 1]: the final *logit* row is dropped, not the final
    text token. A label array of length [text_len - 1] raises rather than
    quietly scoring a prefix of the caption.

    Returns [nan] when the whole caption is padding — nothing to score,
    and 0.0 would be indistinguishable from perfect confidence. *)
let vlm_loss ~(num_vision_tokens : int) ~(text_mask : bool array) ~(labels : int array)
    (logits : mat) : float =
  let shifted = shifted_text_logits ~num_vision_tokens logits in
  let rows = Array.length shifted in
  if Array.length text_mask <> rows || Array.length labels <> rows then
    invalid_arg
      (Printf.sprintf
         "Vlm.vlm_loss: %d shifted rows but mask has %d and labels %d" rows (Array.length text_mask)
         (Array.length labels));
  let targets =
    Array.mapi (fun i keep -> if keep then labels.(i) else Loss.default_ignore_index) text_mask
  in
  Loss.cross_entropy shifted targets

(* ------------------------------------------------------------------ *)
(* Caption generation                                                   *)
(* ------------------------------------------------------------------ *)

(** The sequence a caption starts from: the adapter's output and
    **nothing else**.

    [vlm_generation.py] sets [input_sequence = vision_embeddings] and
    calls the decoder with [input_embedded=True]. Two consequences that
    this reproduces and that [fused_embeddings] above does not:

    - No text prompt. The model's first prediction is conditioned on the
      image alone; the port has no tokenizer, so "no prompt" is also the
      only thing expressible.
    - **No positional embeddings anywhere.** [GPTModel.forward]'s
      `x = x + pos_emb` sits inside `if not input_embedded:`, so the
      decoder adds none. Compare training, where
      [fused_embeddings] uses [Gpt.embed] and each text token carries a
      baked-in position. So the generation path must *not* go through
      [fused_embeddings] or [Gpt.add_pos]: doing so measures 1.36e-01
      against the reference, which is rel ~1e-2 — enough to ruin a
      caption, far too small for any structural check to catch.

    Also worth stating because it looks like an omission: there is no
    separator token between the end of the vision and the start of the
    text. The reference trains without one ("we expect the VLM to start
    generating text tokens right after the vision tokens"), so adding a
    BOS here would be inventing a token the model has never seen. *)
let vision_sequence (m : model) ~(image : Vit.image) : mat = vision_embeddings m ~image

(** One prefill over the image and the first decode step's context.
    Returns the full logits over the vision sequence; the caller reads the
    last row, exactly as [vlm_generation.py] does with [logits[:, -1, :]].

    [~context_length] truncates from the left, matching the reference's
    [input_sequence[:, -context_length:]]. Truncating a cached run would
    be wrong — the cache cannot be un-pushed — so it is asserted rather
    than attempted: a sequence longer than the context is refused instead
    of quietly decoding from a cache that no longer matches the
    reference's window. *)
let prefill_cached ~(context_length : int) (m : model) (cache : Kv_cache.t) ~(image : Vit.image) : mat =
  let seq = vision_sequence m ~image in
  let n_vision = Array.length seq in
  if n_vision <> num_vision_tokens m then
    invalid_arg
      (Printf.sprintf "Vlm.prefill_cached: vision sequence has %d rows, expected %d" n_vision
         (num_vision_tokens m));
  if n_vision > context_length then
    failwith
      (Printf.sprintf
         "Vlm.prefill_cached: image needs %d positions but context_length is %d — the reference \
          would truncate the image away, and a KV cache cannot un-push what it has already seen"
         n_vision context_length);
  Gpt.prefill_from_emb_cached m.gpt cache seq

(** One decode step: append the bare token embedding for [tok] (no
    positional — the decoder places it at the cache length) and return the
    next token's logits.

    The bare embedding is [Gpt.tok_emb] directly rather than [Gpt.embed],
    for the reason [vision_sequence] gives: during generation the
    reference appends [vlm_model.emb_dict(next_token_id)] and lets
    [input_embedded=True] add the position once. Using [Gpt.embed] would
    add it twice. *)
let decode_step_cached (m : model) (cache : Kv_cache.t) ~(tok : int) : vec =
  if tok < 0 || tok >= m.gpt.Gpt.cfg.Config.vocab_size then
    invalid_arg (Printf.sprintf "Vlm.decode_step_cached: token %d out of range" tok);
  let row = Array.copy m.gpt.Gpt.tok_emb.(tok) in
  Gpt.decode_from_emb_cached m.gpt cache [| row |]

(** Greedy/tempered caption generation from an image, with a KV cache.

    Mirrors [vlm_generation.loop] in [vlm_generation.py]: prefill over the
    image, then repeatedly take the last position's logits, sample, and
    feed the sampled token's embedding back.

    Differences from the reference, both forced and both recorded rather
    than hidden:

    - **A cache.** The reference re-runs the whole sequence every step,
      so it is quadratic in the caption length. The cached path is linear
      and must produce the same tokens; gate G-c is exactly that claim.
    - **No tokenizer.** The port returns token ids. Decoding to a string
      is [tokenizer.decode]'s job and the port has no tokenizer — the
      same boundary [Gpt2]'s generation parity already sits at.

    [~on_token] is called after each token is appended and before the EOS
    break, so a caller can observe the stream. [~eos_ids] reproduces the
    stop contract of [Gpt.generate_cached]: the stopping token **is** part
    of the result and nothing is decoded past it.

    [rng] is explicit and [temp] defaults to 0.0 (greedy). The reference
    defaults to [temp=1.0, top_p=0.9], which samples — and sampling is
    not a function of the model, it is a function of a random stream that
    PyTorch and OCaml do not share. Defaulting to greedy makes the
    function deterministic and therefore gate-able, and a caller who wants
    the reference's stochastic defaults passes them plus an [rng] they are
    willing to lose reproducibility to. *)
let generate_cached (rng : Random.State.t) ~(max_gen : int) ?(context_length : int = 1024)
    ?(eos_ids : int list = []) ?(top_k : int option = None) ?(top_p : float option = None)
    ?(min_p : float option = None) ?(temp : float = 0.0) ?on_token (m : model) ~(image : Vit.image) :
    int array =
  if max_gen <= 0 then invalid_arg "Vlm.generate_cached: max_gen must be positive";
  let cache = Kv_cache.create ~num_layers:m.gpt.Gpt.cfg.Config.num_layers in
  (* [cur] is always "the rows whose last row predicts the next token":
     the whole vision sequence after prefill, a single row after each
     decode step. Tracked by value rather than by index because the row
     count changes from n_vision to 1 on the first decode, and an index
     carried across that transition reads out of bounds. *)
  let cur = ref (prefill_cached ~context_length m cache ~image) in
  let last_row (mm : mat) = mm.(Array.length mm - 1) in
  let out = ref [] in
  let pos = ref 0 in
  let finished = ref false in
  while !pos < max_gen && not !finished do
    let next = Sampling.sample ~top_k ~top_p ~min_p ~temp rng (last_row !cur) in
    out := next :: !out;
    (match on_token with Some f -> f !pos next | None -> ());
    if List.mem next eos_ids then finished := true
    else if !pos + 1 < max_gen then (
      (* Only re-run the trunk when another token is wanted: after the
         last one there is nothing to condition on, and pushing it would
         leave the cache longer than the reference's would be. *)
      cur := [| decode_step_cached m cache ~tok:next |]);
    incr pos
  done;
  Array.of_list (List.rev !out)
