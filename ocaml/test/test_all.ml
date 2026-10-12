(** Self-contained test runner (stdlib only, no alcotest/dune needed).
    Mirrors the `if __name__ == "__main__"` checks in
    [gpt_attention.py] and [generate.py].

    Run: [ocamlopt ... test/test_all.ml -o test_all && ./test_all] *)

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

let approx_eq a b eps = abs_float (a -. b) < eps

let () =
  (* --- Tensor.softmax sums to 1 --- *)
  let p = softmax [| 1.0; 2.0; 3.0 |] in
  let s = Array.fold_left ( +. ) 0.0 p in
  check "softmax_sums_to_1" (approx_eq s 1.0 1e-9);
  check "softmax_argmax_is_2" (argmax p = 2);

  (* --- softmax all-masked fallback (left-pad edge case) --- *)
  let u = softmax [| neg_inf; neg_inf |] in
  check "softmax_all_masked_uniform" (approx_eq u.(0) 0.5 1e-9 && approx_eq u.(1) 0.5 1e-9);

  (* --- mat_mul known answer --- *)
  let c = mat_mul [| [| 1.0; 2.0 |]; [| 3.0; 4.0 |] |] [| [| 5.0; 6.0 |]; [| 7.0; 8.0 |] |] in
  check "mat_mul" (approx_eq c.(0).(0) 19.0 1e-9 && approx_eq c.(1).(1) 50.0 1e-9);

  (* --- LayerNorm: zero mean, unit var (unbiased=false, like torch) --- *)
  let ln = Norm.layernorm_make 3 in
  let y = Norm.layernorm_forward ln [| [| 1.0; 2.0; 3.0 |] |] in
  check "layernorm_mean_zero" (approx_eq (mean_vec y.(0)) 0.0 1e-6);
  check "layernorm_std_one" (approx_eq (std_vec y.(0) 0.0) 1.0 1e-3);

  (* --- GELU: exact-erf values --- *)
  check "gelu_zero" (approx_eq (Norm.gelu_scalar 0.0) 0.0 1e-9);
  check "gelu_one" (approx_eq (Norm.gelu_scalar 1.0) 0.8413447461 1e-6);
  check "gelu_neg_one" (approx_eq (Norm.gelu_scalar (-1.0)) (-0.1586552539) 1e-6);
  (* HF "gelu_new" (tanh approx): the constant the exact erf must NOT equal *)
  check "gelu_new_one" (approx_eq (Norm.gelu_new_scalar 1.0) 0.8411919906 1e-9);

  (* --- MHA output shape --- *)
  let rs = Random.State.make [| 123 |] in
  let mha = Attention.mha_make rs ~d_in:4 ~d_out:4 ~num_heads:2 ~ctx_len:8 ~qkv_bias:false () in
  let x = randn_mat rs 3 4 in
  let o = Attention.mha_forward mha x in
  check "mha_shape" (Array.length o = 3 && Array.length o.(0) = 4);

  (* --- Causal property: row 0 of [a;b] forward = forward of [a] alone.
         (Softmax over [s0, -inf] == softmax over [s0].) --- *)
  let rs2 = Random.State.make [| 7 |] in
  let mha2 = Attention.mha_make rs2 ~d_in:4 ~d_out:4 ~num_heads:2 ~ctx_len:8 ~qkv_bias:false () in
  let xa = [| [| 0.43; 0.15; 0.89; 0.10 |] |] in
  let xab = [| [| 0.43; 0.15; 0.89; 0.10 |]; [| 0.55; 0.87; 0.66; 0.20 |] |] in
  let oa = Attention.mha_forward mha2 xa in
  let oab = Attention.mha_forward mha2 xab in
  let maxdiff = ref 0.0 in
  for j = 0 to 3 do
    maxdiff := max !maxdiff (abs_float (oa.(0).(j) -. oab.(0).(j)))
  done;
  check "causal_first_row_stable" (!maxdiff < 1e-9);

  (* --- Padding mask: masked key gets zero weight.
         Single head, craft weights so scores are equal, then mask key 1. --- *)
  let rs3 = Random.State.make [| 99 |] in
  let mha3 = Attention.mha_make rs3 ~d_in:2 ~d_out:2 ~num_heads:1 ~ctx_len:4 ~qkv_bias:false () in
  let x2 = [| [| 1.0; 0.0 |]; [| 0.0; 1.0 |] |] in
  let o_nomask = Attention.mha_forward mha3 x2 in
  let o_mask = Attention.mha_forward ~attn_mask:(Some [| true; false |]) mha3 x2 in
  (* row 1 unmasked mixes both rows; masked row 1 sees only row 0 *)
  let same_as_row0 =
    approx_eq o_mask.(1).(0) o_mask.(0).(0) 1e-9 && approx_eq o_mask.(1).(1) o_mask.(0).(1) 1e-9
  in
  let differs = abs_float (o_nomask.(1).(0) -. o_mask.(1).(0)) > 1e-9 in
  check "padding_mask_redirects_weight" (same_as_row0 && differs);

  (* --- Transformer block shape --- *)
  let rs4 = Random.State.make [| 5 |] in
  let blk = Block.block_make rs4 Config.gpt_tiny_test_config () in
  let xb = randn_mat rs4 4 Config.gpt_tiny_test_config.emb_dim in
  let ob = Block.block_forward blk xb in
  check "block_shape"
    (Array.length ob = 4 && Array.length ob.(0) = Config.gpt_tiny_test_config.emb_dim);

  (* --- GPT forward shapes --- *)
  let rs5 = Random.State.make [| 123 |] in
  let model = Gpt.make rs5 Config.gpt_tiny_test_config () in
  let ids = [| 1; 2; 3 |] in
  let logits = Gpt.forward model ids in
  check "gpt_logits_shape"
    (Array.length logits = 3 && Array.length logits.(0) = Config.gpt_tiny_test_config.vocab_size);
  let last = Gpt.forward_last_only model ids in
  check "gpt_last_only_len" (Array.length last = Config.gpt_tiny_test_config.vocab_size);
  let mask = [| true; true; false |] in
  let last_real = Gpt.forward_last_only ~attn_mask:(Some mask) model ids in
  check "gpt_last_real_matches_row1"
    (approx_eq last_real.(0) logits.(1).(0) 1e-9 && approx_eq last_real.(7) logits.(1).(7) 1e-9);
  Printf.printf "INFO tiny_model_params=%d\n" (Gpt.num_parameters model);

  (* --- Sampling filters --- *)
  let probs = [| 0.5; 0.3; 0.15; 0.05 |] in
  let fk = Sampling.top_k_filter probs 2 in
  check "top_k_keeps_2" (fk.(0) > 0.0 && fk.(1) > 0.0 && fk.(2) = 0.0 && fk.(3) = 0.0);
  let fp = Sampling.top_p_filter probs 0.6 None in
  (* cumsum: 0.5, 0.8 > 0.6 → pivot idx1 kept, rest dropped *)
  check "top_p_pivot" (fp.(0) > 0.0 && fp.(1) > 0.0 && fp.(2) = 0.0 && fp.(3) = 0.0);
  let fm = Sampling.min_p_filter probs 0.5 1 in
  (* thresh = 0.25 → keeps 0.5, 0.3 *)
  check "min_p_thresh" (fm.(0) > 0.0 && fm.(1) > 0.0 && fm.(2) = 0.0);
  let rs6 = Random.State.make [| 1 |] in
  check "greedy_argmax" (Sampling.sample ~temp:0.0 rs6 [| 0.1; 0.9; 0.3 |] = 1);
  let s1 = Sampling.sample ~temp:1.0 ~top_k:(Some 2) rs6 [| 1.0; 2.0; 0.5; 0.1 |] in
  check "sampled_in_topk" (s1 = 0 || s1 = 1);

  (* --- KV cache push/peek --- *)
  let cache = Kv_cache.create ~num_layers:2 in
  check "kv_empty_len" (Kv_cache.past_len cache 0 = 0);
  Kv_cache.push cache ~layer_idx:0
    ~keys:[| [| [| 1.0; 2.0 |] |] |]
    ~values:[| [| [| 3.0; 4.0 |] |] |];
  check "kv_past_len_1" (Kv_cache.past_len cache 0 = 1);
  (match Kv_cache.peek cache 0 with
  | None -> check "kv_peek_some" false
  | Some kv -> check "kv_peek_some" (approx_eq kv.keys.(0).(0).(0) 1.0 1e-9));

  (* --- generate_simple_loop appends max_gen greedy tokens --- *)
  let rs7 = Random.State.make [| 42 |] in
  let tiny = Gpt.make rs7 Config.gpt_tiny_test_config () in
  let out =
    Sampling.generate_simple_loop ~temp:0.0 rs7
      ~model_forward:(Gpt.forward tiny)
      ~context_length:Config.gpt_tiny_test_config.context_length
      [| 1; 2 |] ~max_gen:5
  in
  check "generate_len" (Array.length out = 7);
  check "generate_prefix_kept" (out.(0) = 1 && out.(1) = 2);

  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)
