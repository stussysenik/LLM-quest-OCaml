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
