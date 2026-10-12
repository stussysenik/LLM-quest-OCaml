(** Sampling + generation loop, mirroring [llm_quest/generate.py].

    Ports [sampling], [_top_k_sampling], [_top_p_sampling],
    [_min_p_sampling] and [generate_simple_loop] for the single-sequence
    case. Batched + KV-cache loops ([generate_loop_kv_cache],
    [generate_batched_loop_*]) are the natural next step once a
    batched model forward exists.

    Semantics match Python exactly:
    - [temp = 0.] → greedy argmax;
    - otherwise divide logits by temp, softmax, filter, renormalize,
      multinomial sample;
    - [top_p] and [min_p] are mutually exclusive (assert);
    - nucleus keeps the pivot token via the HF shift-by-one trick. *)

open Tensor

(** Keep only the [k] largest probs, zero the rest. *)
let top_k_filter (probs : vec) (k : int) : vec =
  let n = Array.length probs in
  let k = min k n in
  let idx = Array.init n (fun i -> i) in
  Array.sort (fun a b -> compare probs.(b) probs.(a)) idx;
  let keep = Array.make n false in
  for r = 0 to k - 1 do
    keep.(idx.(r)) <- true
  done;
  Array.mapi (fun i p -> if keep.(i) then p else 0.0) probs

(** Nucleus filter with the pivot-keep trick from HF (see Python).
    Optionally pre-restricted to top-[k] when [top_k] is given. *)
let top_p_filter (probs : vec) (p : float) (top_k : int option) : vec =
  let n = Array.length probs in
  let work = Array.copy probs in
  (match top_k with
  | None -> ()
  | Some k ->
      let idx = Array.init n (fun i -> i) in
      Array.sort (fun a b -> compare work.(b) work.(a)) idx;
      let cutoff = work.(idx.(min k n - 1)) in
      for i = 0 to n - 1 do
        if work.(i) < cutoff then work.(i) <- 0.0
      done);
  let idx = Array.init n (fun i -> i) in
  Array.sort (fun a b -> compare work.(b) work.(a)) idx;
  let sorted = Array.init n (fun r -> work.(idx.(r))) in
  let cum = ref 0.0 in
  let cut = ref n in
  let r = ref 0 in
  while !r < n && !cut = n do
    cum := !cum +. sorted.(!r);
    if !cum > p then cut := !r;
    incr r
  done;
  (* [cut] = first index where cumsum exceeds p = pivot, kept.
     Everything strictly after it is removed (mask shifted by one,
     first token always kept). *)
  let remove = Array.make n false in
  for r = !cut + 1 to n - 1 do
    remove.(r) <- true
  done;
  let out_sorted = Array.mapi (fun r v -> if remove.(r) then 0.0 else v) sorted in
  let out = Array.make n 0.0 in
  for r = 0 to n - 1 do
    out.(idx.(r)) <- out_sorted.(r)
  done;
  out

(** Min-p filter: drop tokens with [p < min_p * p_max], always keeping
    at least [min_tokens_to_keep] (mirrors the paper's arg). *)
let min_p_filter (probs : vec) (min_p : float) (min_tokens_to_keep : int) : vec =
  let n = Array.length probs in
  let p_max = ref probs.(0) in
  for i = 1 to n - 1 do
    if probs.(i) > !p_max then p_max := probs.(i)
  done;
  let thresh = min_p *. !p_max in
  let out = Array.mapi (fun _ p -> if p < thresh then 0.0 else p) probs in
  (* restore top-[min_tokens_to_keep] regardless of threshold *)
  let idx = Array.init n (fun i -> i) in
  Array.sort (fun a b -> compare probs.(b) probs.(a)) idx;
  for r = 0 to min min_tokens_to_keep n - 1 do
    out.(idx.(r)) <- probs.(idx.(r))
  done;
  out

let renormalize (probs : vec) : vec =
  let s = ref 0.0 in
  Array.iter (fun v -> s := !s +. v) probs;
  if !s <= 0.0 then Array.make (Array.length probs) (1.0 /. float_of_int (Array.length probs))
  else Array.map (fun v -> v /. !s) probs

(** Mirrors [sampling()]: greedy at [temp=0.], else tempered sample. *)
let sample ?(top_k : int option = None) ?(top_p : float option = None) ?(min_p : float option = None)
    ~(temp : float) (rs : Random.State.t) (logits : vec) : int =
  (match (top_p, min_p) with
  | Some _, Some _ -> invalid_arg "Sampling.sample: cannot use top_p and min_p together"
  | _ -> ());
  if temp = 0.0 then argmax logits
  else
    let scaled = Array.map (fun v -> v /. temp) logits in
    let probs = ref (softmax scaled) in
    (match (min_p, top_p, top_k) with
    | Some mp, _, _ ->
        let keep = match top_k with None -> 1 | Some k -> k in
        probs := min_p_filter !probs mp keep
    | None, Some p, _ -> probs := top_p_filter !probs p top_k
    | None, None, Some k -> probs := top_k_filter !probs k
    | None, None, None -> ());
    let normed = renormalize !probs in
    multinomial rs normed

(** Mirrors [generate_simple_loop]: repeatedly take the last-position
    logits, sample one token, append. [model_forward] is any
    [ids -> logits (seq x vocab)] function (e.g. [Gpt.forward]). *)
let generate_simple_loop ?(top_k : int option = None) ?(top_p : float option = None)
    ?(min_p : float option = None) ~(temp : float) (rs : Random.State.t)
    ~(model_forward : int array -> mat) ~(context_length : int) (prompt : int array) ~(max_gen : int) :
    int array =
  let ids = ref (Array.copy prompt) in
  for _ = 1 to max_gen do
    let s = Array.length !ids in
    let start = max 0 (s - context_length) in
    let window = Array.init (s - start) (fun i -> !ids.(start + i)) in
    let logits = model_forward window in
    let last = logits.(Array.length logits - 1) in
    let next = sample ~top_k ~top_p ~min_p ~temp rs last in
    ids := Array.append !ids [| next |]
  done;
  !ids
