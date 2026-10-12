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

type linear = { w : mat; b : vec option; d_in : int; d_out : int }

let linear_make (rs : Random.State.t) d_in d_out ~bias : linear =
  (* GPT-2 style init: N(0, 0.02); matches the "from scratch" training path. *)
  let w =
    let m = randn_mat rs d_in d_out in
    Array.map (Array.map (fun v -> v *. 0.02)) m
  in
  let b = if bias then Some (zeros_vec d_out) else None in
  { w; b; d_in; d_out }

let linear_forward_batch (l : linear) (x : mat) : mat = linear_batch x l.w l.b

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

(** Single-head scaled dot-product attention with causal + padding masks.
    [k_all]/[v_all] already include any KV-cache prefix of length [k_len];
    queries cover the last [q_len] rows. *)
let head_forward ~q:(qh : mat) ~k_all ~(v_all : mat) ~(attn_mask : bool array option) : mat =
  let q_len = Array.length qh in
  let k_len = Array.length k_all in
  let hd = if q_len = 0 then 0 else Array.length qh.(0) in
  let scale = 1.0 /. sqrt (float_of_int hd) in
  (* scores = Q @ K^T * scale, shape (q_len x k_len) *)
  let scores = zeros_mat q_len k_len in
  let q_start = k_len - q_len in
  for i = 0 to q_len - 1 do
    for j = 0 to k_len - 1 do
      let s = ref 0.0 in
      for d = 0 to hd - 1 do
        s := !s +. qh.(i).(d) *. k_all.(j).(d)
      done;
      scores.(i).(j) <- !s *. scale
    done
  done;
  (* causal mask: query row i (global pos q_start+i) attends only to j <= pos *)
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
  let weights = softmax_rows scores in
  mat_mul weights v_all

let mha_forward ?(attn_mask : bool array option = None) ?(kv : Kv_cache.layer_kv option = None)
    (a : mha) (x : mat) : mat =
  let s = Array.length x in
  let q = linear_forward_batch a.w_q x in
  let k = linear_forward_batch a.w_k x in
  let v = linear_forward_batch a.w_v x in
  let qh = split_heads q a.num_heads a.head_dim in
  let kh = split_heads k a.num_heads a.head_dim in
  let vh = split_heads v a.num_heads a.head_dim in
  (* Prepend cache prefix if provided (inference with KV cache). *)
  let kh_full =
    match kv with
    | None -> kh
    | Some { keys; _ } -> Array.init a.num_heads (fun h -> Array.append keys.(h) kh.(h))
  in
  let vh_full =
    match kv with
    | None -> vh
    | Some { values; _ } -> Array.init a.num_heads (fun h -> Array.append values.(h) vh.(h))
  in
  let ctx =
    Array.init a.num_heads (fun h -> head_forward ~q:qh.(h) ~k_all:kh_full.(h) ~v_all:vh_full.(h) ~attn_mask)
  in
  let merged = merge_heads ctx in
  ignore s;
  linear_forward_batch a.out_proj merged

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
