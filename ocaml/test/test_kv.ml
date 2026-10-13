(** Gates K1-K5 for the KV-cache generation loop (ROADMAP step 5).

    - K1: greedy decode with cache ≡ without cache, token-for-token
      (the fundamental invariant), plus prefill logits bit-exact.
    - K2: cache K/V after prompt+3 steps ≡ full-forward traces over the
      final ids, per layer/head, bit-exact.
    - K3: batched cache run ≡ single-cache run per item.
    - K4: EOS freeze — item hitting EOS stops (append EOS, freeze),
      the other item is unaffected (≡ its single run).
    - K5: positions — decode step at pos p ≡ full-forward row p
      (absolute positions), and per-item cache counters advance once
      per token even with uneven prompt lengths.

    Run: [make test_kv && ./test_kv] (or [make test] / [make ci]). *)

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

let eq_vec_exact (a : vec) (b : vec) : bool =
  Array.length a = Array.length b && Array.for_all2 ( = ) a b

let () =
  let cfg = Config.gpt_tiny_test_config in
  let model = Gpt.make (Random.State.make [| 31337 |]) cfg () in
  let prompt = [| 4; 11; 7 |] in
  let ctx = (Config.max_positions cfg) in

  (* ---------- K1: cache ≡ no-cache ---------- *)
  let ref_seq =
    Sampling.generate_simple_loop ~temp:0.0 (Random.State.make [| 1 |])
      ~model_forward:(Gpt.forward model) ~context_length:ctx prompt ~max_gen:8
  in
  let kv_seq = Gpt.generate_cached ~temp:0.0 (Random.State.make [| 1 |]) model prompt ~max_gen:8 in
  check "k1_cache_eq_nocache"
    (Array.length kv_seq = Array.length ref_seq && Array.for_all2 ( = ) kv_seq ref_seq);

  let cache0 = Kv_cache.create ~num_layers:cfg.Config.num_layers in
  let pre = Gpt.prefill_cached model cache0 prompt in
  check "k1_prefill_eq_forward" (eq_mat_exact pre (Gpt.forward model prompt));

  (* ---------- K5a: absolute positions ---------- *)
  let ids4 = [| 4; 11; 7; 2 |] in
  let full4 = Gpt.forward model ids4 in
  let cpos = Kv_cache.create ~num_layers:cfg.Config.num_layers in
  ignore (Gpt.prefill_cached model cpos [| 4; 11; 7 |]);
  let row3 = Gpt.decode_step_cached model cpos 2 3 in
  check "k5a_decode_pos_eq_full_row" (eq_vec_exact row3 full4.(3));

  (* ---------- K2: cache state ≡ full-forward traces ---------- *)
  let cache2 = Kv_cache.create ~num_layers:cfg.Config.num_layers in
  let pre2 = Gpt.prefill_cached model cache2 prompt in
  let rs2 = Random.State.make [| 2 |] in
  let ids = ref (Array.copy prompt) in
  let last = ref pre2.(Array.length prompt - 1) in
  for _ = 1 to 3 do
    let nx = Sampling.sample ~temp:0.0 rs2 !last in
    ids := Array.append !ids [| nx |];
    last := Gpt.decode_step_cached model cache2 nx (Array.length !ids - 1)
  done;
  let _, traces = Gpt.forward_with_kv model !ids in
  let k2_ok = ref true in
  List.iteri
    (fun li (kh, vh) ->
      match Kv_cache.peek cache2 li with
      | None -> k2_ok := false
      | Some kv ->
          Array.iter2 (fun a b -> if not (eq_mat_exact a b) then k2_ok := false) kv.Kv_cache.keys kh;
          Array.iter2
            (fun a b -> if not (eq_mat_exact a b) then k2_ok := false)
            kv.Kv_cache.values vh)
    traces;
  check "k2_cache_eq_traces" !k2_ok;

  (* ---------- K3: batched cache ≡ single-cache ---------- *)
  (* Greedy (temp=0) ignores rs entirely, so draw order across items is
     irrelevant: same model + same prompt ⇒ same tokens, exactly. *)
  let prompts = [| [| 4; 11; 7 |]; [| 9; 2 |] |] in
  let batched =
    Gpt.generate_cached_batch ~temp:0.0 (Random.State.make [| 3 |]) model prompts ~max_gen:6
  in
  let single0 = Gpt.generate_cached ~temp:0.0 (Random.State.make [| 99 |]) model prompts.(0) ~max_gen:6 in
  let single1 = Gpt.generate_cached ~temp:0.0 (Random.State.make [| 7 |]) model prompts.(1) ~max_gen:6 in
  check "k3_item0_eq_single"
    (Array.length batched.(0) = Array.length single0 && Array.for_all2 ( = ) batched.(0) single0);
  check "k3_item1_eq_single"
    (Array.length batched.(1) = Array.length single1 && Array.for_all2 ( = ) batched.(1) single1);

  (* ---------- K5b: per-item position counters ---------- *)
  (* Uneven prompts (3 vs 2): three manual decode steps must advance
     each cache by exactly one row per step on EVERY layer, at absolute
     positions len-1, len, len+1 — no shared counter. *)
  let cA = Kv_cache.create ~num_layers:cfg.Config.num_layers in
  let cB = Kv_cache.create ~num_layers:cfg.Config.num_layers in
  ignore (Gpt.prefill_cached model cA prompts.(0));
  ignore (Gpt.prefill_cached model cB prompts.(1));
  for step = 1 to 3 do
    ignore (Gpt.decode_step_cached model cA 1 (Array.length prompts.(0) + step - 1));
    ignore (Gpt.decode_step_cached model cB 2 (Array.length prompts.(1) + step - 1))
  done;
  let counters_ok = ref true in
  for li = 0 to cfg.Config.num_layers - 1 do
    if Kv_cache.past_len cA li <> 6 || Kv_cache.past_len cB li <> 5 then counters_ok := false
  done;
  check "k5b_per_item_counters" !counters_ok;

  (* ---------- K4: EOS freeze ---------- *)
  (* Deterministic EOS pick: run both items fully, choose a token from
     A's run that B's run never emits, then rerun batched with it as
     EOS. A must freeze right after EOS; B must be byte-identical to
     its un-EOS'd single run (no cross-contamination). *)
  let fullA = Gpt.generate_cached ~temp:0.0 (Random.State.make [| 11 |]) model prompts.(0) ~max_gen:8 in
  let fullB = Gpt.generate_cached ~temp:0.0 (Random.State.make [| 11 |]) model prompts.(1) ~max_gen:8 in
  let genA = Array.sub fullA 3 8 in
  let genB = Array.sub fullB 2 8 in
  let eos =
    let rec find i =
      if i >= Array.length genA then failwith "K4: no EOS candidate (adjust seed)";
      if Array.mem genA.(i) genB then find (i + 1) else genA.(i)
    in
    find 0
  in
  let eos_step = let i = ref 0 in (while genA.(!i) <> eos do incr i done; !i) in
  let frozen =
    Gpt.generate_cached_batch ~temp:0.0 ~eos_ids:(Some [ eos ]) (Random.State.make [| 11 |]) model prompts
      ~max_gen:8
  in
  let expectA_len = 3 + eos_step + 1 in
  check "k4_freeze_at_eos"
    (Array.length frozen.(0) = expectA_len && frozen.(0).(expectA_len - 1) = eos);
  check "k4_other_unaffected"
    (Array.length frozen.(1) = Array.length fullB && Array.for_all2 ( = ) frozen.(1) fullB);

  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)
