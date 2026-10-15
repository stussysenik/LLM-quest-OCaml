(** Cross-entropy and accuracy — the two measurements the multimodal
    code takes of itself.

    Ported from [vlm_engine.vlm_loss] and [vit_engine.ViT]'s statics.
    What did **not** come across is the loop: those functions take a
    [DataLoader] and return a number averaged over batches, and this port
    has no dataset, no loader and no shuffling. What is here is the
    per-batch arithmetic they reduce, plus the reduction rule, because
    the reduction rule has two behaviours that are easy to get wrong and
    invisible when wrong:

    - [ignore_index] rows are dropped from **both** numerator and
      denominator. Averaging over all rows instead would silently
      dilute a caption loss with however many padding positions the batch
      happened to have — a number that changes with batch composition
      rather than with the model.
    - an empty reduction is **NaN**, not zero. [calc_loss_loader]
      returns [float("NaN")] for an empty loader, and 0.0 would be
      indistinguishable from a model that is confidently right about
      nothing.

    Numerically, [log_softmax] subtracts the row max before exponentiating.
    Without that, a logit of 1e3 overflows to [inf] and the loss is NaN —
    and only for the rare batch, which is the worst kind of bug. *)

open Tensor

(** Row-wise [log softmax], max-subtracted for stability.

    Returns logs directly rather than a probability vector: the only
    consumers here are cross-entropy and argmax, neither of which needs
    the normalised values, and [exp] of a log-probability is one
    operation cheaper than a full normalisation followed by a [log]. *)
let log_softmax (v : vec) : vec =
  let m = Array.fold_left (fun acc x -> if x > acc then x else acc) neg_infinity v in
  if m = neg_infinity then invalid_arg "Loss.log_softmax: row is all -inf";
  let shifted = Array.map (fun x -> x -. m) v in
  let lse =
    (* log(sum(exp(.))) by the max-subtracted route: stable for any shift *)
    let s = Array.fold_left (fun acc x -> acc +. exp x) 0.0 shifted in
    log s
  in
  Array.map (fun x -> x -. lse) shifted

(** Index of the largest entry; ties resolve to the lowest index, which
    is what [torch.argmax] does. *)
let argmax (v : vec) : int =
  if Array.length v = 0 then invalid_arg "Loss.argmax: empty row";
  let best = ref neg_infinity and bi = ref 0 in
  Array.iteri (fun i x -> if x > !best then (best := x; bi := i)) v;
  !bi

(** The sentinel [F.cross_entropy] and [masked_fill] use for "this row is
    not a target". Named rather than inlined so the two places that agree
    on it ([vlm_loss]'s masking and [cross_entropy]'s filter) cannot drift
    into using different numbers — which would turn every padded caption
    into a hard error. *)
let default_ignore_index = -100

(** Mean cross-entropy over [logits.(i)] against [targets.(i)], skipping
    rows whose target is [ignore_index].

    Returns [nan] when every row is ignored, matching
    [calc_loss_loader]'s empty-loader behaviour. Raises if the row counts
    disagree, because a silent truncat here would change the denominator
    and produce a plausible wrong loss. *)
let cross_entropy ?(ignore_index = default_ignore_index) (logits : mat) (targets : int array) : float =
  let n = Array.length logits in
  if Array.length targets <> n then
    invalid_arg
      (Printf.sprintf "Loss.cross_entropy: %d logit rows but %d targets" n (Array.length targets));
  let total = ref 0.0 and count = ref 0 in
  Array.iteri
    (fun i row ->
      if targets.(i) <> ignore_index then (
        let ls = log_softmax row in
        let t = targets.(i) in
        if t < 0 || t >= Array.length ls then
          invalid_arg
            (Printf.sprintf "Loss.cross_entropy: target %d out of range for row %d of %d" t i
               (Array.length ls));
        (* Cross-entropy is the *negated* mean log-probability.
           [ls.(t)] is already negative, so it is subtracted, not added —
           adding it would report -loss, which passes no comparison and
           would be "close" in magnitude. *)
        total := !total -. ls.(t);
        incr count))
    logits;
  if !count = 0 then nan else !total /. float_of_int !count

(** [correct, total] over the rows whose target is not [ignore_index].
    Returns [(0, 0)] when everything is ignored, so a caller that divides
    by the total gets NaN rather than a division by zero — the same
    "nothing to say" signal the loss uses. *)
let accuracy ?(ignore_index = default_ignore_index) (logits : mat) (targets : int array) : int * int =
  let n = Array.length logits in
  if Array.length targets <> n then
    invalid_arg
      (Printf.sprintf "Loss.accuracy: %d logit rows but %d targets" n (Array.length targets));
  let correct = ref 0 and total = ref 0 in
  Array.iteri
    (fun i row ->
      if targets.(i) <> ignore_index then (
        incr total;
        if argmax row = targets.(i) then incr correct))
    logits;
  (!correct, !total)

(** Running average over batches, with [calc_loss_loader]'s semantics: an
    empty run is NaN.

    Note what this does **not** do, because it is easy to assume it does:
    it is a mean of per-batch means, so it does not weight by example
    count. [calc_loss_loader] returns [total_loss / num_batches], and the
    last batch of a loader is usually short — so the reference's average
    is the average over *batches*, not over examples, and is slightly
    biased toward whatever the tail batch contains.

    That is the reference's behaviour and this mirrors it. Reweighting to
    a true per-example mean would be a different function with a different
    number, and "more correct" is not a reason to disagree with the thing
    being ported. It is recorded here so the choice is visible to whoever
    later wonders why the loss does not quite match a per-example mean. *)
type loss_accum = { mutable sum : float; mutable batches : int }

let acc_create () : loss_accum = { sum = 0.0; batches = 0 }

let acc_add (a : loss_accum) (loss : float) : unit =
  a.sum <- a.sum +. loss;
  a.batches <- a.batches + 1

let acc_mean (a : loss_accum) : float = if a.batches = 0 then nan else a.sum /. float_of_int a.batches

(** Running correct/total, for the same reason. *)
type acc_accum = { mutable correct : int; mutable total : int }

let accuracy_acc_create () : acc_accum = { correct = 0; total = 0 }

let accuracy_acc_add (a : acc_accum) (c : int) (t : int) : unit =
  a.correct <- a.correct + c;
  a.total <- a.total + t

(** [accuracy_loader] returns a *fraction*, which is NaN when nothing was
    seen. Spelled out rather than left to the caller so the empty case has
    one home. *)
let accuracy_acc_frac (a : acc_accum) : float = if a.total = 0 then nan else float_of_int a.correct /. float_of_int a.total
