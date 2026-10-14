(** Vision Transformer (ViT), mirroring
    [llm_quest/multimodal/vision_transformer/] — [vit_model.py]
    ([PatchEmbedding2D], [ViTModel]), [vit_attention.py]
    ([ViTMultiHeadAttention]) and [vit_transformer_block.py]
    ([LayerNorm], [FFN], [ViTTransformerBlock]).

    Multimodal Part 1. The ViT is GPT's decoder with three changes, and it
    is worth naming all three because each one is a place where reusing a
    GPT kernel would be *almost* right:

    1. **Patch embedding replaces token embedding.** An image is cut into
       non-overlapping [patch_size x patch_size] squares and projected to
       [emb_dim]. The Python does it with a [Conv2d] of stride equal to its
       kernel — which is a linear projection of a flattened patch, done
       with a convolution. Ported as the convolution it is written as,
       because a [Conv2d] weight is **[out_channels, in_channels, kh, kw]**
       and *not* this project's [(out, in)] linear layout. That makes it
       the one weight in the repo that does not follow the convention, and
       flattening it at load would be the quiet way to lose a transpose.

    2. **Attention is unmasked.** Every patch attends to every patch: an
       image has no causal structure. See [Attention.mha_forward_encoder].

    3. **A learnable CLS token carries the classification.** After the
       final norm the *first* sequence element — never a mean-pool, never
       the last — is fed to the classifier.

    Plus a fourth difference that is easy to miss and is worth its own
    note: **the LayerNorm convention flips.** ViT's own [LayerNorm] divides
    by [std + eps] — eps *outside* the sqrt — while HuggingFace's, which
    [Norm.layer_norm] defaults to for a loaded checkpoint, divides by
    [sqrt(var + eps)]. They differ by ~5e-6 relative, which is invisible in
    training and fatal for a 1e-9 parity gate. [Norm.layernorm] with
    [Eps_outside_sqrt] is the ViT one; see [layernorm_make].

    Dropout is the identity here, as in the decoder, but *not* because the
    rate is 0 — [VIT_BASE_CONFIG] ships [drop_rate = 0.1]. [nn.Dropout] is
    identity in eval mode whatever its rate, and parity is an inference
    property. (This corrects the blanket "all shipped configs use 0.0"
    note elsewhere in this repo: ViT is the counterexample.)

    Images are [b, num_channels, height, width] in the Python; this port
    takes a single image as [num_channels, height, width] and loops the
    batch explicitly. Batching is wiring, not math — the same reasoning as
    [Tensor.bmat] and [Attention.mha_forward_batch] — and it keeps one
    image's arithmetic on one array. *)

open Tensor

(* ------------------------------------------------------------------ *)
(* Config                                                              *)
(* ------------------------------------------------------------------ *)

(** One image, [num_channels, height, width] — three levels, so it cannot
    be confused with a [Tensor.mat] (which is two). A patch is
    [num_channels, P, P] and is never materialised: [patch_flat] reads it
    straight out of the image, because the conv's kernel is indexed in that
    channel-major order anyway. *)
type image = Tensor.mat array

type t = {
  img_height : int;
  img_width : int;
  patch_size : int;
  num_channels : int;
  emb_dim : int;
  n_layers : int;
  n_heads : int;
  qkv_bias : bool;
  num_classes : int;
}

(** Number of patches, [H*W / P^2]. Checked rather than assumed: a
    non-integer is how a wrong [patch_size] announces itself here instead
    of as a silent misalignment between the conv's output grid and the
    positional table. *)
let num_patches (c : t) : int = (c.img_height * c.img_width) / (c.patch_size * c.patch_size)

let num_patches_h (c : t) : int = c.img_height / c.patch_size

let num_patches_w (c : t) : int = c.img_width / c.patch_size

(** [VIT_BASE_CONFIG] from [config.py]: 224x224 RGB, 16x16 patches,
    emb 768, 12 layers / 12 heads, [qkv_bias = True], 100 classes. *)
let vit_base_config =
  {
    img_height = 224;
    img_width = 224;
    patch_size = 16;
    num_channels = 3;
    emb_dim = 768;
    n_layers = 12;
    n_heads = 12;
    qkv_bias = true;
    num_classes = 100;
  }

(* ------------------------------------------------------------------ *)
(* Weights                                                             *)
(* ------------------------------------------------------------------ *)

(** The patch projection, straight out of [nn.Conv2d] and deliberately
    **not** reshaped into the project convention: [conv_weight] is
    [emb_dim] kernels of [num_channels * P^2] entries each, i.e. the
    [(emb_dim, num_channels, P, P)] tensor with its last two axes
    flattened, and [conv_bias] is [emb_dim].

    Conv2d is the one weight in this project outside the [(out, in)]
    linear convention. Flattening it to [(emb_dim, num_channels*P^2)] at
    load would be fine *arithmetically* and would destroy the gate that
    notices it was done to the wrong axis — which is silent, because
    [emb_dim] and [num_channels * P^2] are 768 and 768 for
    [VIT_BASE_CONFIG]. The square case again. *)
type patch_conv = { conv_weight : Tensor.vec array; conv_bias : Tensor.vec; cls_token : Tensor.vec }

(** One encoder block. [ln_1]/[ln_2] are [Norm.layernorm]s built with
    [Eps_outside_sqrt]; [att] is an [Attention.mha]; [ffn] is a plain
    two-layer MLP with biases and exact GELU — GPT's FFN, *not* SwiGLU. *)
type block = {
  ln_1 : Norm.layernorm;
  ln_2 : Norm.layernorm;
  att : Attention.mha;
  ffn_l1 : Attention.linear;
  ffn_l2 : Attention.linear;
}

type model = {
  cfg : t;
  patch : patch_conv;
  pos_embedding : Tensor.mat; (** [1, num_patches + 1, emb_dim] — includes the CLS slot *)
  blocks : block array;
  final_ln : Norm.layernorm;
  classifier : Attention.linear;
}

(* ------------------------------------------------------------------ *)
(* Patch embedding                                                     *)
(* ------------------------------------------------------------------ *)

(** One patch, flattened channel-major: index [c * P * P + i * P + j] is
    channel [c], row [i], col [j] within the patch.

    This ordering is what makes the conv weight's
    [(emb_dim, num_channels, P, P)] layout line up with the dot product
    below, and it is the same order [Conv2d] indexes its kernel in. *)
let patch_flat (img : image) ~num_channels ~patch_size ~top ~left : Tensor.vec =
  let p2 = patch_size * patch_size in
  Array.init (num_channels * p2) (fun k ->
      let c = k / p2 in
      let r = (k mod p2) / patch_size in
      let cc = k mod patch_size in
      img.(c).(top + r).(left + cc))

(** [patch_embedding img] → [(num_patches + 1, emb_dim)]: the [num_patches]
    projected patches followed by the CLS token at position 0.

    Patch order is **row-major over the (H/P, W/P) grid**, i.e. patch
    [(r, c)] is at index [r * num_patches_w + c]. That order is not
    arbitrary: the positional table is indexed against it, so a
    row-major/column-major mix-up produces a self-consistent model that
    learns to compensate and matches nothing. Gate V-c pins it.

    The CLS token is prepended *after* the patches, matching the Python's
    [torch.cat([cls_tokens, x], dim=1)] — it is sequence position 0. *)
let patch_embedding (m : model) (img : image) : Tensor.mat =
  let c = m.cfg in
  let emb = c.emb_dim in
  let p = c.patch_size in
  let nph = num_patches_h c in
  let npw = num_patches_w c in
  if Array.length img <> c.num_channels then
    invalid_arg
      (Printf.sprintf "Vit.patch_embedding: image has %d channels, expected %d" (Array.length img)
         c.num_channels);
  (* patches, row-major *)
  let patches =
    Array.init (nph * npw) (fun p_idx ->
        let r = p_idx / npw in
        let col = p_idx mod npw in
        let flat = patch_flat img ~num_channels:c.num_channels ~patch_size:p ~top:(r * p) ~left:(col * p) in
        Array.init emb (fun o ->
            let w = m.patch.conv_weight.(o) in
            let s = ref 0.0 in
            for k = 0 to Array.length flat - 1 do
              s := !s +. w.(k) *. flat.(k)
            done;
            !s +. m.patch.conv_bias.(o)))
  in
  (* CLS token first, then the patches *)
  Array.init (1 + (nph * npw)) (fun i -> if i = 0 then Array.copy m.patch.cls_token else patches.(i - 1))

(* ------------------------------------------------------------------ *)
(* Block                                                               *)
(* ------------------------------------------------------------------ *)

(** The FFN: [Linear(emb, 4*emb)] → exact-erf GELU → [Linear(4*emb, emb)],
    both with biases. Same shape as GPT's, and a different op from Llama's
    SwiGLU — a ViT is a GPT decoder underneath. *)
let ffn_forward (b : block) (x : mat) : mat =
  let h = Norm.apply_activation_mat Norm.Gelu_exact (Attention.linear_forward_batch b.ffn_l1 x) in
  Attention.linear_forward_batch b.ffn_l2 h

(** Pre-norm encoder block: LayerNorm → attention → residual, then
    LayerNorm → FFN → residual. Same pre-norm shape as GPT's block, with
    unmasked attention in place of the causal kind. *)
let block_forward (b : block) (x : mat) : mat =
  let a1 = Attention.mha_forward_encoder b.att (Norm.layer_norm b.ln_1 x) in
  let x1 = Tensor.add x a1 in
  let a2 = ffn_forward b (Norm.layer_norm b.ln_2 x1) in
  Tensor.add x1 a2

(* ------------------------------------------------------------------ *)
(* Forward                                                             *)
(* ------------------------------------------------------------------ *)

(** [forward_hidden_states m img] → the final-normed sequence,
    [(num_patches + 1, emb_dim)]. *)
let forward_hidden_states (m : model) (img : image) : Tensor.mat =
  let x = patch_embedding m img in
  let x = Tensor.add x m.pos_embedding in
  let x = Array.fold_left (fun acc b -> block_forward b acc) x m.blocks in
  Norm.layer_norm m.final_ln x

(** [forward m img] → [(num_classes)], the classifier applied to the
    **CLS token** — sequence position 0 of the final-normed state. Not a
    mean-pool over patches and not the last position: [ViTModel.forward]
    reads [x[:, 0]], and the CLS token is precisely the learned
    aggregation of the whole image. *)
let forward (m : model) (img : image) : Tensor.vec =
  let hidden = forward_hidden_states m img in
  Attention.linear_forward_batch m.classifier [| hidden.(0) |] |>
  fun rows -> rows.(0)

(* ------------------------------------------------------------------ *)
(* Construction from scratch (for the self-consistency gates)          *)
(* ------------------------------------------------------------------ *)

(** GPT-style init, [N(0, 0.02)], matching [Attention.linear_make] so that
    the from-scratch model and a loaded one are built the same way. *)
let linear_make (rs : Random.State.t) d_in d_out ~bias : Attention.linear =
  let w = randn_mat rs d_in d_out |> Array.map (Array.map (fun v -> v *. 0.02)) |> Tensor.transpose in
  { Attention.w; b = (if bias then Some (zeros_vec d_out) else None); d_in; d_out }

let model_make (rs : Random.State.t) (c : t) : model =
  if c.img_height mod c.patch_size <> 0 || c.img_width mod c.patch_size <> 0 then
    invalid_arg "Vit.model_make: image dims must be divisible by patch_size";
  if c.emb_dim mod c.n_heads <> 0 then invalid_arg "Vit.model_make: emb_dim must be divisible by n_heads";
  let emb = c.emb_dim in
  let ffn_hidden = 4 * emb in
  let k = c.num_channels * c.patch_size * c.patch_size in
  let patch =
    {
      (* One kernel vector per output channel, channel-major over the
         patch — the order [patch_flat] reads back out. *)
      conv_weight = Array.init emb (fun _ -> (randn_mat rs 1 k).(0) |> Array.map (fun v -> v *. 0.02));
      conv_bias = zeros_vec emb;
      cls_token = Array.init emb (fun _ -> (Random.State.float rs 2.0) -. 1.0);
    }
  in
  let mha_make blk_emb = Attention.mha_make rs ~d_in:blk_emb ~d_out:blk_emb ~num_heads:c.n_heads
                            ~ctx_len:(num_patches c + 1) ~qkv_bias:c.qkv_bias ~out_proj_bias:true () in
  {
    cfg = c;
    patch;
    pos_embedding =
      Array.init (num_patches c + 1) (fun _ ->
          Array.init emb (fun _ -> (Random.State.float rs 2.0) -. 1.0));
    blocks =
      Array.init c.n_layers (fun _ ->
          {
            ln_1 = Norm.layernorm_make ~eps:1e-5 emb;
            ln_2 = Norm.layernorm_make ~eps:1e-5 emb;
            att = mha_make emb;
            ffn_l1 = linear_make rs emb ffn_hidden ~bias:true;
            ffn_l2 = linear_make rs ffn_hidden emb ~bias:true;
          });
    final_ln = Norm.layernorm_make ~eps:1e-5 emb;
    classifier = linear_make rs emb c.num_classes ~bias:true;
  }

(** A deterministic random image, [num_channels, height, width], values in
    [-1, 1). Pixels are normalized this way in the training engine too;
    the exact range does not matter for parity, only that it is fixed. *)
let random_image (rs : Random.State.t) (c : t) : image =
  Array.init c.num_channels (fun _ ->
      Array.init c.img_height (fun _ ->
          Array.init c.img_width (fun _ -> Random.State.float rs 2.0 -. 1.0)))