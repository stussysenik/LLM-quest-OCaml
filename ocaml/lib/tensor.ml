(** Minimal dense-tensor helpers mirroring the torch ops used in
    [llm_quest/gpt/gpt_attention.py] and [gpt_transformer_block.py].

    Conventions (row-major, matching PyTorch semantics):
    - [vec] is a 1D row: [float array]
    - [mat] is 2D [seq x dim]: [float array array], i.e. [m.(i).(j)]
    - Linear weights are stored as (d_in x d_out) so that a row-vector
      forward is [x @ w + b], exactly like [nn.Linear] applied to the
      last dim. *)

type vec = float array
type mat = float array array

let neg_inf = -1e30

let zeros_vec n : vec = Array.make n 0.0

let ones_vec n : vec = Array.make n 1.0

let zeros_mat rows cols : mat = Array.make_matrix rows cols 0.0

let copy_mat (m : mat) : mat = Array.map Array.copy m

(** Uniform [0,1) matrix, mirrors [torch.rand]. *)
let rand_mat (rs : Random.State.t) rows cols : mat =
  Array.init rows (fun _ -> Array.init cols (fun _ -> Random.State.float rs 1.0))

(** Standard-normal matrix via Box-Muller, for GPT-style init. *)
let randn_mat (rs : Random.State.t) rows cols : mat =
  let gauss () =
    let u1 = max (Random.State.float rs 1.0) 1e-12 in
    let u2 = Random.State.float rs 1.0 in
    sqrt (-2.0 *. log u1) *. cos (2.0 *. Float.pi *. u2)
  in
  Array.init rows (fun _ -> Array.init cols (fun _ -> gauss ()))

let transpose (m : mat) : mat =
  let r = Array.length m in
  if r = 0 then [||]
  else
    let c = Array.length m.(0) in
    Array.init c (fun j -> Array.init r (fun i -> m.(i).(j)))

(** [a (m x k) @ b (k x n) = (m x n)]. *)
let mat_mul (a : mat) (b : mat) : mat =
  let m = Array.length a in
  let k = if m = 0 then 0 else Array.length a.(0) in
  let kb = Array.length b in
  let n = if kb = 0 then 0 else Array.length b.(0) in
  if k <> kb then invalid_arg "Tensor.mat_mul: inner dims mismatch";
  Array.init m (fun i ->
      Array.init n (fun j ->
          let s = ref 0.0 in
          for l = 0 to k - 1 do
            s := !s +. a.(i).(l) *. b.(l).(j)
          done;
          !s))

(** Row-vector [x (d_in)] @ [w (d_in x d_out)] = [vec (d_out)]. *)
let rowvec_mul (x : vec) (w : mat) : vec =
  let d_in = Array.length x in
  let wr = Array.length w in
  if wr <> d_in then invalid_arg "Tensor.rowvec_mul: dim mismatch";
  let d_out = if wr = 0 then 0 else Array.length w.(0) in
  Array.init d_out (fun j ->
      let s = ref 0.0 in
      for i = 0 to d_in - 1 do
        s := !s +. x.(i) *. w.(i).(j)
      done;
      !s)

(** [linear_fwd (w (d_out x d_in)) (x (s x d_in)) (b : vec option)] =
    [x @ w^T + b] — the one linear op in this repo.

    **[w] is [(out, in)]**, [nn.Linear]'s own layout, which is what HF
    stores on disk for every plain linear here; GPT-2's [Conv1D] is
    [(in, out)] and is transposed once, at load, by
    [Gpt2_loader]. One convention means no call site chooses.

    The dangerous case is the square one: for GPT-2 and for
    tiny-random-Llama, [hidden = heads * head_dim], so every attention
    projection is square and a transposed weight multiplies to *numbers*
    rather than raising. [mat_mul]'s inner-dimension check catches every
    non-square projection; the square ones are covered by the layout gate
    ([layout_gpt2_out_in] / [layout_llama_out_in]), which re-derives the
    expected [out] from the config instead of trusting the loader. *)
let linear_fwd (w : mat) (x : mat) (b : vec option) : mat =
  let y = mat_mul x (transpose w) in
  (match b with
  | None -> ()
  | Some bv ->
      let s = Array.length y in
      let d = if s = 0 then 0 else Array.length y.(0) in
      if Array.length bv <> d then invalid_arg "Tensor.linear_fwd: bias dim mismatch";
      for i = 0 to s - 1 do
        for j = 0 to d - 1 do
          y.(i).(j) <- y.(i).(j) +. bv.(j)
        done
      done);
  y

(** Numerically stable softmax over a vector. If every entry is masked to
    [neg_inf] (the left-padding edge case noted in [gpt_attention.py]),
    falls back to a uniform distribution instead of NaN. *)
let softmax (x : vec) : vec =
  let n = Array.length x in
  let m = ref x.(0) in
  for i = 1 to n - 1 do
    if x.(i) > !m then m := x.(i)
  done;
  let y = Array.map (fun v -> exp (v -. !m)) x in
  let s = ref 0.0 in
  Array.iter (fun v -> s := !s +. v) y;
  if !s = 0.0 || Float.is_nan !s then Array.make n (1.0 /. float_of_int n)
  else (
    for i = 0 to n - 1 do
      y.(i) <- y.(i) /. !s
    done;
    y)

let softmax_rows (m : mat) : mat = Array.map softmax m

let mean_vec (x : vec) : float =
  let s = ref 0.0 in
  Array.iter (fun v -> s := !s +. v) x;
  !s /. float_of_int (Array.length x)

(** Population stddev (divides by N), matching [torch.std(unbiased=False)]. *)
let std_vec (x : vec) (mean : float) : float =
  let s = ref 0.0 in
  Array.iter (fun v -> s := !s +. ((v -. mean) *. (v -. mean))) x;
  sqrt (!s /. float_of_int (Array.length x))

let argmax (x : vec) : int =
  let bi = ref 0 in
  for i = 1 to Array.length x - 1 do
    if x.(i) > x.(!bi) then bi := i
  done;
  !bi

(** Sample one index from a normalized distribution via inverse CDF. *)
let multinomial (rs : Random.State.t) (p : vec) : int =
  let u = Random.State.float rs 1.0 in
  let acc = ref 0.0 in
  let n = Array.length p in
  let i = ref 0 in
  while !i < n - 1 && !acc +. p.(!i) < u do
    acc := !acc +. p.(!i);
    incr i
  done;
  !i

(** Batched sequences: one [(s x d)] token matrix per item. Items in a
    batch share a physical length (shorter ones are padded); see
    [Gpt.forward_batch] for masks and positions.

    Deliberate design: batched ops delegate item-wise to the verified
    single-sequence kernels above. Batching is wiring (padding,
    per-sequence positions, per-sequence readout) — never new math —
    so Gate 1 (batch=1 ≡ single path) holds by construction and the
    gates that bite are 3-5. *)
type bmat = mat array

let bmat_map (f : mat -> mat) (b : bmat) : bmat = Array.map f b

let bmat_mapi (f : int -> mat -> mat) (b : bmat) : bmat = Array.mapi f b
