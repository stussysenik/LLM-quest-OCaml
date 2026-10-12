(** GPT-2 model, mirroring [llm_quest/gpt/gpt_model.py].

    - Token + learned positional embeddings ([emb_dict]/[pos_emb_dict]).
    - Stack of pre-LN transformer blocks ([ModuleList] in Python).
    - Final LayerNorm + tied-off linear head to [vocab_size] (no bias).
    - [forward] maps token ids [(seq)] to logits [(seq x vocab)].
    - [forward_last_only] projects only the last real token
      (the [last_token_only] + [attn_mask] path in Python), which
      avoids a full [seq x vocab] projection during generation. *)

open Tensor

type t = {
  tok_emb : mat;
  pos_emb : mat;
  blocks : Block.block list;
  final_ln : Norm.layernorm;
  lm_head : Attention.linear;
  cfg : Config.gpt_config;
}

let make (rs : Random.State.t) (cfg : Config.gpt_config) ?(activation = Norm.Gelu_exact)
    ?(out_proj_bias = true) () : t =
  let emb m = Array.map (Array.map (fun v -> v *. 0.02)) m in
  {
    tok_emb = emb (randn_mat rs cfg.vocab_size cfg.emb_dim);
    pos_emb = emb (randn_mat rs cfg.context_length cfg.emb_dim);
    blocks = List.init cfg.n_layers (fun _ -> Block.block_make rs cfg ~activation ~out_proj_bias ());
    final_ln = Norm.layernorm_make cfg.emb_dim;
    lm_head = Attention.linear_make rs cfg.emb_dim cfg.vocab_size ~bias:false;
    cfg;
  }

let embed (m : t) (ids : int array) : mat =
  let s = Array.length ids in
  Array.init s (fun i ->
      if ids.(i) < 0 || ids.(i) >= m.cfg.vocab_size then
        invalid_arg "Gpt.embed: token id out of range";
      let te = Array.copy m.tok_emb.(ids.(i)) in
      let pe = m.pos_emb.(i) in
      for j = 0 to m.cfg.emb_dim - 1 do
        te.(j) <- te.(j) +. pe.(j)
      done;
      te)

let forward ?(attn_mask : bool array option = None) (m : t) (ids : int array) : mat =
  let s = Array.length ids in
  if s > m.cfg.context_length then invalid_arg "Gpt.forward: sequence exceeds context_length";
  let x = ref (embed m ids) in
  List.iter (fun blk -> x := Block.block_forward ~attn_mask blk !x) m.blocks;
  let h = Norm.layernorm_forward m.final_ln !x in
  Attention.linear_forward_batch m.lm_head h

(** Per-stage hidden states: [embeddings; block_0; ...; block_{L-1}].
    Mirrors HF `output_hidden_states=True` (13 entries for gpt2):
    entry 0 is the post-dropout embedding (identity in eval), entry
    [i+1] is the output of transformer block [i], pre-[ln_f]. *)
let forward_states ?(attn_mask : bool array option = None) (m : t) (ids : int array) : mat list =
  let s = Array.length ids in
  if s > m.cfg.context_length then invalid_arg "Gpt.forward_states: sequence exceeds context_length";
  let x = ref (embed m ids) in
  let states = ref [ !x ] in
  List.iter
    (fun blk ->
      x := Block.block_forward ~attn_mask blk !x;
      states := !x :: !states)
    m.blocks;
  List.rev !states

(** Logits for the last real token only. [attn_mask] marks real tokens;
    with [None] the physical last row is used. *)
let forward_last_only ?(attn_mask : bool array option = None) (m : t) (ids : int array) : vec =
  let logits = forward ~attn_mask m ids in
  let idx =
    match attn_mask with
    | None -> Array.length ids - 1
    | Some mk ->
        let last = ref (-1) in
        Array.iteri (fun i real -> if real then last := i) mk;
        if !last < 0 then invalid_arg "Gpt.forward_last_only: empty mask";
        !last
  in
  logits.(idx)

let num_parameters (m : t) : int =
  let mcount (a : mat) = Array.length a * if Array.length a = 0 then 0 else Array.length a.(0) in
  let lcount (l : Attention.linear) =
    mcount l.Attention.w + match l.Attention.b with None -> 0 | Some b -> Array.length b
  in
  mcount m.tok_emb + mcount m.pos_emb
  + List.fold_left
      (fun acc (b : Block.block) ->
        acc
        + lcount b.Block.att.Attention.w_q
        + lcount b.Block.att.Attention.w_k
        + lcount b.Block.att.Attention.w_v
        + lcount b.Block.att.Attention.out_proj
        + Array.length b.Block.ln1.Norm.scale * 2
        + Array.length b.Block.ln2.Norm.scale * 2
        + lcount b.Block.ffn.Block.l1
        + lcount b.Block.ffn.Block.l2)
      0 m.blocks
  + Array.length m.final_ln.Norm.scale * 2
  + lcount m.lm_head
