(** Gates 1-5 for the batched forward (see [ocaml/ROADMAP.md] step 4).

    - Gate 1: batch=1 ≡ single-sequence forward (bit-exact: shared kernels).
    - Gate 2: batch=N repeats ≡ N identical outputs (no broadcast bugs).
    - Gate 3: different-length sequences + padding. Real-token rows must
      equal the unpadded single run; changing the pad *value* must not
      move real rows (causal + padding masks compose, neither replaces
      the other). Right-pad aligns positions; left-pad needs explicit
      [pos_ids] so each sequence starts at 0.
    - Gate 4: per-sequence last-token readout. The naive [:, -1, :] read
      is asserted WRONG on padded input (sensitivity proof), the
      per-sequence read is asserted equal to the unpadded single run.
    - Gate 5: greedy decode via batched forward (batch=1) ≡ single loop,
      token for token.

    Run: [make test_batched && ./test_batched] (or [make test]). *)

open Tensor

let passed = ref 0
let failed = ref 0

let check name cond =
  if cond then (
    incr passed;
    Printf.printf "PASS %s\n" name)
  else (
    incr failed;
    Printf.printf "FAIL %s\n" name)

let eq_mat_exact (a : mat) (b : mat) : bool =
  Array.length a = Array.length b
  && Array.for_all2
       (fun ra rb -> Array.length ra = Array.length rb && Array.for_all2 ( = ) ra rb)
       a b

let max_abs_diff (a : mat) (b : mat) : float =
  let m = ref 0.0 in
  Array.iter2
    (fun ra rb -> Array.iter2 (fun x y -> m := max !m (abs_float (x -. y))) ra rb)
    a b;
  !m

let () =
  let rs = Random.State.make [| 2024 |] in
  let cfg = Config.gpt_tiny_test_config in
  let model = Gpt.make rs cfg () in

  (* ---------- Gate 1: batch=1 ≡ single ---------- *)
  let ids = [| 5; 7; 3 |] in
  let single = Gpt.forward model ids in
  let batched = Gpt.forward_batch model [| ids |] in
  check "g1_batch1_eq_single" (Array.length batched = 1 && eq_mat_exact batched.(0) single);

  (* ---------- Gate 2: repeats identical ---------- *)
  let rep = Gpt.forward_batch model [| ids; ids; ids |] in
  check "g2_repeats_identical"
    (eq_mat_exact rep.(0) single && eq_mat_exact rep.(1) single && eq_mat_exact rep.(2) single);

  (* ---------- Gate 3a: right padding ---------- *)
  let short = [| 9; 2 |] in
  let short_single = Gpt.forward model short in
  (* physical rows align with real positions under right padding *)
  let rp_ids = [| [| 5; 7; 3 |]; [| 9; 2; 0 |] |] in
  let rp_mask = [| [| true; true; true |]; [| true; true; false |] |] in
  let rp_out = Gpt.forward_batch ~attn_mask:(Some rp_mask) model rp_ids in
  let item1_real = [| rp_out.(1).(0); rp_out.(1).(1) |] in
  check "g3a_rightpad_real_rows" (eq_mat_exact item1_real short_single);

  (* ---------- Gate 3b: left padding needs pos_ids ---------- *)
  let lp_ids = [| [| 0; 0; 9; 2 |] |] in
  let lp_mask = [| [| false; false; true; true |] |] in
  let lp_pos = [| [| 0; 0; 0; 1 |] |] in
  let lp_out = Gpt.forward_batch ~attn_mask:(Some lp_mask) ~pos_ids:(Some lp_pos) model lp_ids in
  let lp_real = [| lp_out.(0).(2); lp_out.(0).(3) |] in
  check "g3b_leftpad_pos_ids" (eq_mat_exact lp_real short_single);

  (* ---------- Gate 3c: pad VALUE must not leak into real rows ---------- *)
  let rp_ids_other_pad = [| [| 5; 7; 3 |]; [| 9; 2; 63 |] |] in
  let rp_out2 = Gpt.forward_batch ~attn_mask:(Some rp_mask) model rp_ids_other_pad in
  check "g3c_pad_value_hidden"
    (eq_mat_exact rp_out.(0) rp_out2.(0)
    && Array.for_all2 ( = ) rp_out.(1).(0) rp_out2.(1).(0)
    && Array.for_all2 ( = ) rp_out.(1).(1) rp_out2.(1).(1));

  (* ---------- Gate 4: per-sequence last token ---------- *)
  let last_batch = Gpt.forward_last_only_batch ~attn_mask:(Some rp_mask) model rp_ids in
  let last0 = Gpt.forward_last_only model [| 5; 7; 3 |] in
  let last1 = Gpt.forward_last_only model short in
  check "g4_last_token_per_sequence"
    (Array.for_all2 ( = ) last_batch.(0) last0 && Array.for_all2 ( = ) last_batch.(1) last1);
  (* sensitivity: the naive physical-last read IS wrong for the short item *)
  let naive = rp_out.(1).(2) in
  let mx = ref 0.0 in
  Array.iter2 (fun x y -> mx := max !mx (abs_float (x -. y))) naive last1;
  check "g4_naive_last_is_wrong" (!mx > 1e-6);

  (* ---------- Gate 5: greedy decode parity ---------- *)
  let rs1 = Random.State.make [| 77 |] in
  let rs2 = Random.State.make [| 77 |] in
  let prompt = [| 4; 11 |] in
  let seq_single =
    Sampling.generate_simple_loop ~temp:0.0 rs1
      ~model_forward:(Gpt.forward model)
      ~context_length:cfg.Config.context_length prompt ~max_gen:8
  in
  let seq_batched =
    Sampling.generate_simple_loop_batched ~temp:0.0 rs2
      ~model_forward_batch:(Gpt.forward_batch model)
      ~context_length:cfg.Config.context_length [| prompt |] ~max_gen:8
  in
  check "g5_greedy_decode_parity"
    (Array.length seq_batched = 1
    && Array.length seq_batched.(0) = Array.length seq_single
    && Array.for_all2 ( = ) seq_batched.(0) seq_single);
  (* batch=2 shape sanity *)
  let two =
    Sampling.generate_simple_loop_batched ~temp:0.0 (Random.State.make [| 1 |])
      ~model_forward_batch:(Gpt.forward_batch model)
      ~context_length:cfg.Config.context_length
      [| [| 4; 11 |]; [| 9; 2; 5 |] |]
      ~max_gen:4
  in
  check "g5_batch2_lengths" (Array.length two = 2 && Array.length two.(0) = 6 && Array.length two.(1) = 7);

  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)
