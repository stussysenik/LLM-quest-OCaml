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

let embed_with_pos (m : t) (ids : int array) (pos : int array) : mat =
  let s = Array.length ids in
  if s > m.cfg.context_length then invalid_arg "Gpt.embed: sequence exceeds context_length";
  if Array.length pos <> s then invalid_arg "Gpt.embed: pos_ids length mismatch";
  Array.init s (fun i ->
      if ids.(i) < 0 || ids.(i) >= m.cfg.vocab_size then invalid_arg "Gpt.embed: token id out of range";
      if pos.(i) < 0 || pos.(i) >= m.cfg.context_length then
        invalid_arg "Gpt.embed: position id out of range";
      let te = Array.copy m.tok_emb.(ids.(i)) in
      let pe = m.pos_emb.(pos.(i)) in
      for j = 0 to m.cfg.emb_dim - 1 do
        te.(j) <- te.(j) +. pe.(j)
      done;
      te)

let embed (m : t) (ids : int array) : mat =
  embed_with_pos m ids (Array.init (Array.length ids) (fun j -> j))

(** Position ids per item: [pos_ids.(i).(j)] is the positional-embedding
    row for physical row [j] of item [i]. Default (None) is identity —
    correct for unpadded and right-padded batches, where physical and
    real positions coincide. Left-padded items pass HF-style ids
    (pads → 0, real tokens counted from 0), so every sequence's first
    real token starts at position 0 (Gate 3). *)
let embed_batch ?(pos_ids : int array array option = None) (m : t) (bids : int array array) : mat array
    =
  Array.mapi
    (fun i ids ->
      let pos =
        match pos_ids with None -> Array.init (Array.length ids) (fun j -> j) | Some ps -> ps.(i)
      in
      embed_with_pos m ids pos)
    bids

let mask_of_batch (attn_mask : bool array array option) (i : int) : bool array option =
  match attn_mask with None -> None | Some ms -> Some ms.(i)

(** Shared trunk: blocks + final LN + head, on pre-embedded input. *)
let forward_from_emb ?(attn_mask : bool array option = None) (m : t) (emb : mat) : mat =
  let x = ref emb in
  List.iter (fun blk -> x := Block.block_forward ~attn_mask blk !x) m.blocks;
  let h = Norm.layernorm_forward m.final_ln !x in
  Attention.linear_forward_batch m.lm_head h

let forward ?(attn_mask : bool array option = None) (m : t) (ids : int array) : mat =
  forward_from_emb ~attn_mask m (embed m ids)

(** Batched forward (Gates 1-3): one trunk run per item with its own
    padding mask. Causal + padding masks compose inside [mha_forward];
    neither replaces the other. *)
let forward_batch ?(attn_mask : bool array array option = None) ?(pos_ids : int array array option = None)
    (m : t) (bids : int array array) : mat array =
  let embs = embed_batch ~pos_ids m bids in
  Array.mapi (fun i e -> forward_from_emb ~attn_mask:(mask_of_batch attn_mask i) m e) embs

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

(** Last real-token index of a padding mask ([-1] if empty). *)
let last_real_index (mk : bool array) : int =
  let last = ref (-1) in
  Array.iteri (fun i real -> if real then last := i) mk;
  !last

(** Batched last-token readout (Gate 4): logits at [len_i - 1] per item,
    NOT at the physical last position. Reading [:, -1, :] on a padded
    batch silently returns pad-row logits for short sequences — the
    single most likely batching bug, pinned by gate 4. *)
let forward_last_only_batch ?(attn_mask : bool array array option = None)
    ?(pos_ids : int array array option = None) (m : t) (bids : int array array) : vec array =
  let all_logits = forward_batch ~attn_mask ~pos_ids m bids in
  Array.mapi
    (fun i lg ->
      let idx =
        match attn_mask with
        | None -> Array.length bids.(i) - 1
        | Some ms -> (
            match last_real_index ms.(i) with
            | -1 -> invalid_arg "Gpt.forward_last_only_batch: empty mask"
            | k -> k)
      in
      lg.(idx))
    all_logits

(** Full forward that also returns per-layer (K, V) head matrices — the
    exact values a cache must hold after seeing [ids] (Gate K2). *)
let forward_with_kv ?(attn_mask : bool array option = None) (m : t) (ids : int array) :
    mat * (Tensor.mat array * Tensor.mat array) list =
  let x = ref (embed m ids) in
  let traces = ref [] in
  List.iter
    (fun blk ->
      let out, kh, vh = Block.block_forward_kv ~attn_mask blk !x in
      traces := (kh, vh) :: !traces;
      x := out)
    m.blocks;
  let h = Norm.layernorm_forward m.final_ln !x in
  (Attention.linear_forward_batch m.lm_head h, List.rev !traces)

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

(* ------------------------------------------------------------------ *)
(* KV-cache generation (ROADMAP step 5, Gates K1-K5).                   *)
(* ------------------------------------------------------------------ *)

(** Prefill: full forward over the prompt, pushing each layer's K/V into
    [cache] (which must be fresh). Returns full logits; the last row
    seeds the first decode step. *)
let prefill_cached (m : t) (cache : Kv_cache.t) ?(attn_mask : bool array option = None)
    (ids : int array) : mat =
  if Kv_cache.past_len cache 0 <> 0 then invalid_arg "Gpt.prefill_cached: cache must be fresh";
  let x = ref (embed m ids) in
  List.iteri
    (fun li blk ->
      let kv = Kv_cache.peek cache li in
      let out, kh, vh = Block.block_forward_kv ~attn_mask ~kv blk !x in
      Kv_cache.push cache ~layer_idx:li ~keys:kh ~values:vh;
      x := out)
    m.blocks;
  let h = Norm.layernorm_forward m.final_ln !x in
  Attention.linear_forward_batch m.lm_head h

(** One decode step: single token [tok] at absolute position [pos],
    attending over the cache prefix plus itself.

    Mask shape is [1, N+1] all-ones: every cached key is valid history
    and the new key is the only new entry. The causal mask is implicit
    in the cache here — recomputing a triangular mask would either drop
    the prefix (wrong) or double-count the current token (wrong). *)
let decode_step_cached (m : t) (cache : Kv_cache.t) (tok : int) (pos : int) : vec =
  let x = ref (embed_with_pos m [| tok |] [| pos |]) in
  List.iteri
    (fun li blk ->
      let kv = Kv_cache.peek cache li in
      let mask = Array.make (Kv_cache.past_len cache li + 1) true in
      let out, kh, vh = Block.block_forward_kv ~attn_mask:(Some mask) ~kv blk !x in
      Kv_cache.push cache ~layer_idx:li ~keys:kh ~values:vh;
      x := out)
    m.blocks;
  let h = Norm.layernorm_forward m.final_ln !x in
  (Attention.linear_forward_batch m.lm_head h).(0)

(** Greedy/tempered generation with KV cache (Gate K1): prefill once,
    then one decode step per token. Must reproduce
    [Sampling.generate_simple_loop] token-for-token. *)
let generate_cached ?(top_k : int option = None) ?(top_p : float option = None)
    ?(min_p : float option = None) ~(temp : float) (rs : Random.State.t) (m : t) (prompt : int array)
    ~(max_gen : int) : int array =
  let cache = Kv_cache.create ~num_layers:(List.length m.blocks) in
  let logits = prefill_cached m cache prompt in
  let ids = ref (Array.copy prompt) in
  let last = ref logits.(Array.length logits - 1) in
  for _ = 1 to max_gen do
    let next = Sampling.sample ~top_k ~top_p ~min_p ~temp rs !last in
    ids := Array.append !ids [| next |];
    last := decode_step_cached m cache next (Array.length !ids - 1)
  done;
  !ids

(** Batched cache generation (Gates K3-K5): one cache per item, one
    batched step at a time. Positions are per-sequence counters
    ([len_i - 1] after each append, Gate K5). Finished items (in
    [eos_ids]) freeze: no sampling, no append, no cache push — so a
    finished sequence can neither grow nor corrupt the others (Gate
    K4). Without [eos_ids] every item runs the full [max_gen]. *)
let generate_cached_batch ?(top_k : int option = None) ?(top_p : float option = None)
    ?(min_p : float option = None) ?(eos_ids : int list option = None) ~(temp : float)
    (rs : Random.State.t) (m : t) (prompts : int array array) ~(max_gen : int) : int array array =
  let n = Array.length prompts in
  let caches = Array.init n (fun _ -> Kv_cache.create ~num_layers:(List.length m.blocks)) in
  let ids = ref (Array.map Array.copy prompts) in
  let lasts = Array.init n (fun i -> (prefill_cached m caches.(i) prompts.(i)).(Array.length prompts.(i) - 1)) in
  let finished = Array.make n false in
  let is_eos tok = match eos_ids with None -> false | Some es -> List.mem tok es in
  for _ = 1 to max_gen do
    if Array.for_all (fun f -> f) finished then ()
    else
      for i = 0 to n - 1 do
        if not finished.(i) then (
          let next = Sampling.sample ~top_k ~top_p ~min_p ~temp rs lasts.(i) in
          (* every sampled token — EOS included — is appended, matching
             the repo's generate loops; only non-EOS tokens advance the
             cache, so finished items freeze instead of corrupting. *)
          ids := Array.mapi (fun j seq -> if j = i then Array.append seq [| next |] else seq) !ids;
          if is_eos next then finished.(i) <- true
          else lasts.(i) <- decode_step_cached m caches.(i) next (Array.length !ids.(i) - 1))
      done
  done;
  !ids
