(** Causal self-attention, mirroring [llm_quest/gpt/gpt_attention.py].

    Progression kept from the Python file:
    - [SelfAttention_v1]: three raw rand matrices, no mask.
    - [SelfAttention_v2]: linear (w + optional bias) projections, no mask.
    - [SelfAttention_v3]: + causal mask (single batch for clarity).
    - [mha]: optimized multi-head attention with head splitting,
      causal mask, padding mask ([attn_mask]: true = real token),
      and an optional hook to prepend KV-cache K/V
      (see [Kv_cache]). Dropout is a no-op at inference
      ([drop_rate = 0.0] in every shipped config).

    Shapes below use [s] = seq len, [d] = emb dim, [h] = heads. *)

open Tensor

(** A linear layer's weights. [w] is **[(out, in)]** — [d_out] rows,
    [d_in] columns — matching [Tensor.linear_fwd] and [nn.Linear]. See
    that function for why the square-projection case needs a gate and
    not just a type. *)
type linear = { w : mat; b : vec option; d_in : int; d_out : int }

let linear_make (rs : Random.State.t) d_in d_out ~bias : linear =
  (* GPT-2 style init: N(0, 0.02); matches the "from scratch" training path.
     Drawn as [(in, out)] then transposed rather than drawn as
     [(out, in)] directly: transpose only permutes the floats, so every
     from-scratch model keeps the exact numbers it had before the
     layout convention changed. That leaves the convention refactor with
     a total "nothing moved" result instead of "only the loaded-weights
     parity stayed put". *)
  let w =
    let m = randn_mat rs d_in d_out in
    Array.map (Array.map (fun v -> v *. 0.02)) m |> Tensor.transpose
  in
  let b = if bias then Some (zeros_vec d_out) else None in
  { w; b; d_in; d_out }

let linear_forward_batch (l : linear) (x : mat) : mat = linear_fwd l.w x l.b

type mha = {
  w_q : linear;
  w_k : linear;
  w_v : linear;
  out_proj : linear;
  num_heads : int;
  head_dim : int;
  d_out : int;
  ctx_len : int;
}

let mha_make (rs : Random.State.t) ~d_in ~d_out ~num_heads ~ctx_len ~qkv_bias ?(out_proj_bias = true) ()
    : mha =
  if d_out mod num_heads <> 0 then invalid_arg "Attention.mha_make: d_out must be divisible by num_heads";
  {
    w_q = linear_make rs d_in d_out ~bias:qkv_bias;
    w_k = linear_make rs d_in d_out ~bias:qkv_bias;
    w_v = linear_make rs d_in d_out ~bias:qkv_bias;
    out_proj = linear_make rs d_out d_out ~bias:out_proj_bias;
    num_heads;
    head_dim = d_out / num_heads;
    d_out;
    ctx_len;
  }

(** Split [x (s x d_out)] into per-head [h x s x head_dim]. *)
let split_heads (x : mat) num_heads head_dim : mat array =
  let s = Array.length x in
  Array.init num_heads (fun h ->
      Array.init s (fun i -> Array.init head_dim (fun d -> x.(i).(h * head_dim + d))))

(** Merge [h x s x head_dim] back to [(s x d_out)]. *)
let merge_heads (hs : mat array) : mat =
  let h = Array.length hs in
  let s = if h = 0 then 0 else Array.length hs.(0) in
  let hd = if h = 0 || s = 0 then 0 else Array.length hs.(0).(0) in
  Array.init s (fun i ->
      Array.init (h * hd) (fun j ->
          let head = j / hd in
          let d = j mod hd in
          hs.(head).(i).(d)))

(** Scaled dot-product scores [Q @ K^T / sqrt(head_dim)], shape
    [(q_len x k_len)]. Queries cover the last [q_len] rows, so the causal
    boundary is [q_start = k_len - q_len].

    Split out from [head_forward] so that the decoder and the encoder share
    it verbatim. The arithmetic and its ordering are unchanged — this is a
    move, not a rewrite, and GPT-2's parity numbers are the check that it
    stayed one. *)
let head_scores ~(q : mat) ~(k_all : mat) : mat =
  let q_len = Array.length q in
  let k_len = Array.length k_all in
  let hd = if q_len = 0 then 0 else Array.length q.(0) in
  let scale = 1.0 /. sqrt (float_of_int hd) in
  let scores = zeros_mat q_len k_len in
  for i = 0 to q_len - 1 do
    for j = 0 to k_len - 1 do
      let s = ref 0.0 in
      for d = 0 to hd - 1 do
        s := !s +. q.(i).(d) *. k_all.(j).(d)
      done;
      scores.(i).(j) <- !s *. scale
    done
  done;
  scores

(** Single-head scaled dot-product attention with causal + padding masks.
    [k_all]/[v_all] already include any KV-cache prefix of length [k_len];
    queries cover the last [q_len] rows. *)
let head_forward ~q:(qh : mat) ~k_all ~(v_all : mat) ~(attn_mask : bool array option) : mat =
  let q_len = Array.length qh in
  let k_len = Array.length k_all in
  let scores = head_scores ~q:qh ~k_all in
  (* causal mask: query row i (global pos q_start+i) attends only to j <= pos *)
  let q_start = k_len - q_len in
  for i = 0 to q_len - 1 do
    let pos = q_start + i in
    for j = 0 to k_len - 1 do
      let causal_hide = j > pos in
      let pad_hide =
        match attn_mask with
        | None -> false
        | Some m -> if j < Array.length m then not m.(j) else false
      in
      if causal_hide || pad_hide then scores.(i).(j) <- neg_inf
    done
  done;
  mat_mul (softmax_rows scores) v_all

(** Single-head attention with **no mask at all** — the encoder form.

    This is the only difference between a decoder and a ViT's attention:
    every patch attends to every patch, because an image has no causal
    structure. Dropping the causal mask is also what makes the KV cache
    meaningless here, which is why there is no [~kv] on the encoder path
    rather than a cache that is quietly ignored. *)
let head_forward_encoder ~q:(qh : mat) ~k_all ~(v_all : mat) : mat =
  mat_mul (softmax_rows (head_scores ~q:qh ~k_all)) v_all

(** Split stage: Q/K/V projections, split into per-head matrices.
    Exposed so the KV-cache loop can capture the current K/V
    (see [Block.block_forward_kv], [Gpt.prefill_cached]). *)
let project_split (a : mha) (x : mat) : mat array * mat array * mat array =
  let q = linear_forward_batch a.w_q x in
  let k = linear_forward_batch a.w_k x in
  let v = linear_forward_batch a.w_v x in
  ( split_heads q a.num_heads a.head_dim,
    split_heads k a.num_heads a.head_dim,
    split_heads v a.num_heads a.head_dim )

(** Prefix stage: prepend the cache-held K/V rows (if any) to the
    current ones. Pure concatenation — no math. *)
let with_prefix (a : mha) (kv : Kv_cache.layer_kv option) (kh : mat array) (vh : mat array) :
    mat array * mat array =
  match kv with
  | None -> (kh, vh)
  | Some { Kv_cache.keys; values } ->
      ( Array.init a.num_heads (fun h -> Array.append keys.(h) kh.(h)),
        Array.init a.num_heads (fun h -> Array.append values.(h) vh.(h)) )

(** Combine stage: per-head causal+padding attention, merge, output
    projection. *)
let combine (a : mha) ?(attn_mask : bool array option = None) (qh : mat array) (kh_full : mat array)
    (vh_full : mat array) : mat =
  let ctx =
    Array.init a.num_heads (fun h ->
        head_forward ~q:qh.(h) ~k_all:kh_full.(h) ~v_all:vh_full.(h) ~attn_mask)
  in
  linear_forward_batch a.out_proj (merge_heads ctx)

let mha_forward ?(attn_mask : bool array option = None) ?(kv : Kv_cache.layer_kv option = None)
    (a : mha) (x : mat) : mat =
  let qh, kh, vh = project_split a x in
  let kh_full, vh_full = with_prefix a kv kh vh in
  combine ~attn_mask a qh kh_full vh_full

(** Teaching helper: unmasked single-head attention (v1/v2 behaviour). *)
let single_head_no_mask ~w_q ~w_k ~w_v (x : mat) : mat =
  let q = linear_forward_batch w_q x in
  let k = linear_forward_batch w_k x in
  let v = linear_forward_batch w_v x in
  let hd = if Array.length q = 0 then 1 else Array.length q.(0) in
  let scale = 1.0 /. sqrt (float_of_int hd) in
  let scores = mat_mul q (transpose k) in
  Array.iter (fun row -> Array.iteri (fun j s -> row.(j) <- s *. scale) row) scores;
  mat_mul (softmax_rows scores) v

(** Batched MHA (parity Gate 1-2): item [i] runs the verified
    [mha_forward] with causal + padding mask [attn_mask.(i)]
    ([true] = real token). Padding keys are hidden; causal structure
    within the real tokens is unchanged (Gate 3 pins this). *)
let mha_forward_batch ?(attn_mask : bool array array option = None) (a : mha) (xs : mat array) :
    mat array =
  Array.mapi
    (fun i x ->
      let m = match attn_mask with None -> None | Some ms -> Some ms.(i) in
      mha_forward ~attn_mask:m a x)
    xs

(** Encoder MHA — [ViTMultiHeadAttention] in
    [llm_quest/multimodal/vision_transformer/vit_attention.py]. Structurally
    [mha_forward] with the causal mask removed and no padding mask: image
    patches are a fixed WxH grid, so there is no autoregressive structure
    and no padding to mask.

    It reuses the same [mha] record and the same [head_scores], so the two
    paths differ *only* in the masking — which is the property gate V-a
    checks by running the decoder form with every position visible and
    requiring it to agree. The record is reused rather than duplicated
    because the weights are identical in shape and convention; a separate
    type would imply a difference that is not there. *)
let mha_forward_encoder (a : mha) (x : mat) : mat =
  let qh, kh, vh = project_split a x in
  let ctx =
    Array.init a.num_heads (fun h -> head_forward_encoder ~q:qh.(h) ~k_all:kh.(h) ~v_all:vh.(h))
  in
  linear_forward_batch a.out_proj (merge_heads ctx)

(* ------------------------------------------------------------------ *)
(* Grouped-Query Attention — [GroupedQueryAttention] in                *)
(* [llm_quest/gpt_to_llama3/llama_attention.py] (step D5).             *)
(* ------------------------------------------------------------------ *)

(** GQA differs from MHA in one place: K and V are projected onto
    [num_kv_groups] heads instead of [num_heads], and each K/V head then
    serves [num_repeat = num_heads / num_kv_groups] query heads. The
    projection *shapes* carry the difference — [w_k]/[w_v] are
    [(num_kv_groups * head_dim, d_in)], not [(d_out, d_in)] — which is
    why this is a separate type rather than an [mha] with a flag: the
    cache, the loader and the block stack all need to know the head count
    before any math happens.

    [out_proj] keeps a bias and the three projections do not, exactly as
    the Python builds them ([nn.Linear(..., bias=False)] for Q/K/V,
    default bias for [out_proj]). Llama has no biases on Q/K/V —
    [Config.qkv_bias] is [false] — and that is a fact about the
    architecture, not a convention a call site chooses. *)

type gqa = {
  w_q : linear;
  w_k : linear;
  w_v : linear;
  out_proj : linear;
  num_heads : int;
  num_kv_groups : int;
  head_dim : int;
  d_out : int;
}

let num_repeat (a : gqa) : int = a.num_heads / a.num_kv_groups

let gqa_make (rs : Random.State.t) ~d_in ~d_out ~num_heads ~num_kv_groups ?(out_proj_bias = true) ()
    : gqa =
  if d_out mod num_heads <> 0 then invalid_arg "Attention.gqa_make: d_out must be divisible by num_heads";
  if num_heads mod num_kv_groups <> 0 then
    invalid_arg "Attention.gqa_make: num_heads must be divisible by num_kv_groups";
  let head_dim = d_out / num_heads in
  {
    w_q = linear_make rs d_in d_out ~bias:false;
    w_k = linear_make rs d_in (num_kv_groups * head_dim) ~bias:false;
    w_v = linear_make rs d_in (num_kv_groups * head_dim) ~bias:false;
    out_proj = linear_make rs d_out d_out ~bias:out_proj_bias;
    num_heads;
    num_kv_groups;
    head_dim;
    d_out;
  }

(** [repeat_interleave] over the head axis: query head [h] reads K/V head
    [h / num_repeat]. Not [h mod num_kv_groups] — the Python groups are
    *contiguous* blocks ([Q1..Q5] share [K1]), so the mapping is a
    division. Gate D5b is what pins this: the transposed mapping is also a
    permutation, and it produces plausible numbers rather than raising. *)
let repeat_kv_heads (a : gqa) (hs : mat array) : mat array =
  let nr = num_repeat a in
  if Array.length hs <> a.num_kv_groups then
    invalid_arg "Attention.repeat_kv_heads: wrong number of kv heads";
  if nr = 1 then hs else Array.init a.num_heads (fun h -> hs.(h / nr))

(** Project + RoPE + split, mirroring [GroupedQueryAttention.forward]'s
    first half. Q comes back with [num_heads] heads; K and V with
    [num_kv_groups] — the cache stores *these*, which is what makes the
    cache [kv_heads]-sized rather than [num_heads]-sized.

    [positions] is [None] for prefill (positions [0 .. seq_len-1]) and
    [Some ps] for a decode step at arbitrary positions; Q and K share
    them, V is not rotated at all. *)
let gqa_project ?(positions : int array option = None) ~cos ~sin (a : gqa) (x : mat) :
    mat array * mat array * mat array =
  let q = linear_forward_batch a.w_q x in
  let k = linear_forward_batch a.w_k x in
  let v = linear_forward_batch a.w_v x in
  let qh = split_heads q a.num_heads a.head_dim in
  let kh = split_heads k a.num_kv_groups a.head_dim in
  let vh = split_heads v a.num_kv_groups a.head_dim in
  (Array.map (Rope.apply ~positions ~cos ~sin) qh,
   Array.map (Rope.apply ~positions ~cos ~sin) kh,
   vh)

(** Combine stage: expand K/V from [num_kv_groups] to [num_heads], then
    run the *same* [head_forward] the MHA path uses, merge, project.
    Reusing the MHA kernel rather than writing a GQA one is deliberate —
    once the head expansion has happened the two are the same function,
    and a second copy could drift. *)
let gqa_combine ?(attn_mask : bool array option = None) (a : gqa) (qh : mat array)
    (kh : mat array) (vh : mat array) : mat =
  let k_full = repeat_kv_heads a kh in
  let v_full = repeat_kv_heads a vh in
  let ctx =
    Array.init a.num_heads (fun h ->
        head_forward ~q:qh.(h) ~k_all:k_full.(h) ~v_all:v_full.(h) ~attn_mask)
  in
  linear_forward_batch a.out_proj (merge_heads ctx)

let gqa_forward ?(attn_mask : bool array option = None) ?(kv : Kv_cache.layer_kv option = None)
    ?(positions : int array option = None) ~cos ~sin (a : gqa) (x : mat) : mat =
  let qh, kh, vh = gqa_project ~positions ~cos ~sin a x in
  let kh_full, vh_full =
    match kv with
    | None -> (kh, vh)
    | Some { Kv_cache.keys; values } ->
        if Array.length keys <> a.num_kv_groups then
          invalid_arg
            "Attention.gqa_forward: KV cache holds the wrong head count (must be num_kv_groups, not num_heads)";
        ( Array.init a.num_kv_groups (fun g -> Array.append keys.(g) kh.(g)),
          Array.init a.num_kv_groups (fun g -> Array.append values.(g) vh.(g)) )
  in
  gqa_combine ~attn_mask a qh kh_full vh_full
