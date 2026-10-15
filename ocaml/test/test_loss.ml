(* Gates for [Loss] and the two eval entry points.
   18 gates. The subject is not that softmax is hard — it is
   `ignore_index`, which is invisible until a caption has padding. *)

open Tensor

let passed = ref 0
let failed = ref 0

let check name ok =
  if ok then (
    incr passed;
    Printf.printf "PASS %s\n" name)
  else (
    incr failed;
    Printf.printf "FAIL %s\n" name)

let near a b = abs_float (a -. b) <= 1e-9

let read_case f name =
  let logits = Safetensors.mat_of_f64 f (name ^ ".logits") in
  let targets =
    Safetensors.i64_array_of f (name ^ ".targets") |> Array.map Int64.to_int
  in
  (logits, targets, (Safetensors.vec_of_f64 f (name ^ ".loss")).(0))

let () =
  if Array.length Sys.argv <> 2 then (
    Printf.eprintf "usage: %s <loss_ref.safetensors>\n" Sys.argv.(0);
    exit 2);
  let f = Safetensors.open_file Sys.argv.(1) in
  Printf.printf "Loss gates (cross-entropy with ignore_index, accuracy)\n\n";

  let cases = [ ("L-a", "no rows ignored (the bug is invisible here by construction)");
                ("L-b", "half the rows ignored — the case that discriminates");
                ("L-c", "every row ignored — must be NaN, not 0.0");
                ("L-d", "an exact argmax tie");
                ("L-e", "a single row, so a stray divisor shows up") ] in
  List.iter
    (fun (name, why) ->
      Printf.printf "%s: %s\n" name why;
      let logits, targets, ref_loss = read_case f name in
      let ref_correct = Int64.to_int (Safetensors.i64_array_of f (name ^ ".correct")).(0) in
      let ref_total = Int64.to_int (Safetensors.i64_array_of f (name ^ ".total")).(0) in
      let ref_acc = (Safetensors.vec_of_f64 f (name ^ ".accuracy")).(0) in
      let ours = Loss.cross_entropy logits targets in
      let is_nan x = x <> x in
      if ref_total = 0 then (
        check (name ^ " loss is NaN when nothing is kept") (is_nan ours);
        check (name ^ " accuracy is NaN when nothing is kept")
          (let c, t = Loss.accuracy logits targets in
           c = 0 && t = 0 && is_nan (Loss.accuracy_acc_frac (Loss.accuracy_acc_create ()))))
      else (
        Printf.printf "  %-44s ours=%.12f ref=%.12f\n" (name ^ " cross-entropy") ours ref_loss;
        check (name ^ " cross-entropy matches torch") (near ours ref_loss);
        let c, t = Loss.accuracy logits targets in
        check
          (Printf.sprintf "%s accuracy %d/%d matches torch %d/%d" name c t ref_correct ref_total)
          (c = ref_correct && t = ref_total);
        let acc = Loss.accuracy_acc_create () in
        Loss.accuracy_acc_add acc c t;
        check (name ^ " accuracy fraction matches torch") (near (Loss.accuracy_acc_frac acc) ref_acc)))
    cases;

  Printf.printf "\nL-b the anti-gate: ignoring rows must change the mean\n";
  let logits_b, targets_b, ref_loss_b = read_case f "L-b" in
  (* What a mean-over-all-rows implementation would report: the same
     numerators, divided by the total row count instead of the kept count.
     Here that is exactly half, so no tolerance setting can hide it. *)
  let mean_over_all =
    let total = ref 0.0 and n = ref 0 in
    Array.iteri
      (fun i row ->
        (* The bug in one place: every row counts toward the denominator,
           but an ignored row is simply missing from the numerator. *)
        incr n;
        if targets_b.(i) <> Loss.default_ignore_index then (
          let ls = Loss.log_softmax row in
          total := !total -. ls.(targets_b.(i))))
      logits_b;
    !total /. float_of_int !n
  in
  Printf.printf "  %-44s wrong=%.12f ref=%.12f\n" "L-b mean-over-all-rows" mean_over_all ref_loss_b;
  check "L-b averaging over all rows is a different number" (not (near mean_over_all ref_loss_b));
  check "L-b ignoring rows is a different number"
    (mean_over_all < ref_loss_b -. 1e-6 || mean_over_all > ref_loss_b +. 1e-6);

  Printf.printf "\nL-f structural refusals\n";
  check "L-f row/target count mismatch raises"
    (try
       (* 2 logit rows, 1 target — a genuine mismatch. *)
       ignore (Loss.cross_entropy [| [| 1.0; 0.0 |]; [| 1.0; 0.0 |] |] [| 0 |]);
       false
     with Invalid_argument _ -> true);
  check "L-f a target outside the class range raises"
    (try
       ignore (Loss.cross_entropy [| [| 1.0; 0.0 |] |] [| 7 |]);
       false
     with Invalid_argument _ -> true);
  check "L-f an all -inf row raises rather than returning NaN silently"
    (try
       ignore (Loss.log_softmax [| neg_infinity; neg_infinity |]);
       false
     with Invalid_argument _ -> true);

  Printf.printf "\nL-g log_softmax properties\n";
  let v = [| 1000.0; 1001.0; 999.0 |] in
  let ls = Loss.log_softmax v in
  let sum_exp = Array.fold_left (fun a x -> a +. exp x) 0.0 ls in
  check "L-g log_softmax rows sum to 1 in probability space"
    (abs_float (sum_exp -. 1.0) < 1e-12);
  check "L-g log_softmax survives a logit of 1e3 (no overflow to inf)"
    (Array.for_all (fun x -> x = x && abs_float x < 1e300) ls);
  let l0 = Loss.log_softmax [| 0.0; 0.0 |] in
  check "L-g two equal logits get equal log-probabilities" (near l0.(0) l0.(1) && near l0.(0) (-. (log 2.0)));
  check "L-g argmax of equal entries returns the lowest index"
    (Loss.argmax [| 1.0; 3.0; 3.0; 2.0 |] = 1);

  Printf.printf "\nL-h the accumulators (calc_loss_loader's reduction rule)\n";
  let a = Loss.acc_create () in
  check "L-h an empty accumulator is NaN, not 0.0" (Loss.acc_mean a <> Loss.acc_mean a);
  Loss.acc_add a 2.0;
  Loss.acc_add a 4.0;
  check "L-h a two-batch accumulator is the mean" (near (Loss.acc_mean a) 3.0);
  (* A mean of per-batch means over unequal batches is the bug; a running
     sum over a running count is not. *)
  let b = Loss.acc_create () in
  Loss.acc_add b 1.0;
  Loss.acc_add b 1.0;
  Loss.acc_add b 9.0;
  (* This is a mean of per-batch means, which is what calc_loss_loader
     computes and therefore what this mirrors — it does NOT reweight by
     example count. Stated as a positive gate so the behaviour is pinned
     rather than accidental. *)
  check "L-h the mean is over batches, as calc_loss_loader does" (near (Loss.acc_mean b) (11.0 /. 3.0));
  let c = Loss.acc_create () in
  Loss.acc_add c 5.0;
  check "L-h a single accumulated batch is that batch's value" (near (Loss.acc_mean c) 5.0);

  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)
