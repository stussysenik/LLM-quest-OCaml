(** End-to-end generation parity (ROADMAP step 6): our greedy loops vs
    HuggingFace [generate(do_sample=False)], token-for-token.

    Steps 1-5 prove the forward pass and the KV cache match PyTorch
    numerically. This proves the *loop*: load a real checkpoint, tokenize,
    generate, land on the same token IDs in the same order. Two
    independent OCaml loops are gated, because they fail differently —
    the simple loop re-runs the whole prefix every step, the cached loop
    never does:

    - gen_cached_eq_hf  — [Gpt.generate_cached], all 20 reference tokens
    - gen_simple_eq_hf  — [Sampling.generate_simple_loop], 5-token prefix
      of the same reference

    Token counts differ on purpose. The simple loop projects every
    position to the full 50257-wide vocab on every step, so it costs
    ~1.3s per position against the cached loop's ~1.4s per *token*; 20
    steps of it is ~10 minutes of CI. Both are deterministic greedy runs
    over the same weights, so a 5-token prefix match plus a full 20-token
    cached match covers the same logic at 1/20th the wall clock.

    The gate is exact, and that is safe rather than lucky: the export
    prints the top1-top2 logit margin at every step, and the minimum over
    these 20 steps is 0.036 — two orders of magnitude above the ~1e-4
    fp32 noise of a 12-layer fp32 forward.

    On stopping. HF's stop contract (verified, not assumed: a tiny model
    forced to predict EOS returns prompt + 1 token, EOS included) is
    "append the stopping token, then end". Real gpt2 greedy never emits
    EOS, so that contract cannot be reached from a real prompt — 17
    prompts x 80 steps produced zero EOS hits. Two gates stand in:

    - hf_eos_contract — from HF's own [forced_eos_token_id] output: the
      forced run equals the plain run with only its last token replaced,
      same length. No padding, no continuation past the stop. This is the
      reference-side half of the contract.
    - stop_cached / stop_simple — our half, exercised with a stop token
      the model really produces: output ends ON that token and stops
      there. Without the check the loops would happily continue.

    Token IDs are compared, not decoded text: the OCaml side has no
    tokenizer, by design. The reference file carries the tokenized prompt
    ([gen.prompt_ids]) so both sides start from identical input.

    Usage: [./parity_gen <gpt2_weights.safetensors> <gpt2_gen_ref.safetensors>]
    (second file written by [ocaml/parity/export_gen.py]). *)

let passed = ref 0
let failed = ref 0

let check name cond =
  if cond then (
    incr passed;
    Printf.printf "PASS %s\n" name)
  else (
    incr failed;
    Printf.printf "FAIL %s\n" name)

let ids_str (a : int array) =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int a)) ^ "]"

(** Compare the generated suffix (our loops return prompt ++ generation). *)
let check_ids name ~(prompt : int array) ~(ours : int array) ~(ref_ids : int array) =
  let p = Array.length prompt in
  let gen = Array.sub ours p (Array.length ours - p) in
  let n = min (Array.length gen) (Array.length ref_ids) in
  let d = ref (-1) in
  let i = ref 0 in
  while !i < n && !d < 0 do
    if gen.(!i) <> ref_ids.(!i) then d := !i;
    incr i
  done;
  if !d < 0 && Array.length gen <> Array.length ref_ids then d := n;
  let ok = !d < 0 in
  check name ok;
  if not ok then (
    Printf.printf "    lengths ours=%d ref=%d\n" (Array.length gen) (Array.length ref_ids);
    if !d < Array.length gen && !d < Array.length ref_ids then
      Printf.printf "    first divergence at step %d: ours=%d ref=%d\n" !d gen.(!d) ref_ids.(!d);
    Printf.printf "    ours full: %s\n" (ids_str gen);
    Printf.printf "    ref  full: %s\n" (ids_str ref_ids))

let i64_ints (f : Safetensors.file) (name : string) : int array =
  Array.map Int64.to_int (Safetensors.i64_array_of f name)

(** Last index <= [hi] whose token does not occur earlier, so stopping on
    it is unambiguous (an earlier repeat would stop the run too soon). *)
let fresh_token_at (ids : int array) (hi : int) : int =
  let rec scan k =
    if k < 0 then failwith "Parity_gen: no unambiguous stop token in range";
    let tok = ids.(k) in
    let earlier = ref false in
    for j = 0 to k - 1 do
      if ids.(j) = tok then earlier := true
    done;
    if !earlier then scan (k - 1) else tok
  in
  scan hi

let () =
  if Array.length Sys.argv <> 3 then (
    Printf.eprintf "usage: %s <weights.safetensors> <gen_ref.safetensors>\n" Sys.argv.(0);
    exit 2);
  let model = Hf_loader.load_gpt2 Sys.argv.(1) in
  let genf = Safetensors.open_file Sys.argv.(2) in
  let prompt = i64_ints genf "gen.prompt_ids" in
  let new_ids = i64_ints genf "gen.new_ids" in
  let eos_ids = i64_ints genf "gen.eos_ids" in
  let n = Array.length new_ids in
  (* The forced-EOS run ends on the stopping token, so the id is
     self-describing — no extra field to keep in sync. *)
  let eos = eos_ids.(Array.length eos_ids - 1) in
  let simple_n = 5 in
  let ctx = model.Gpt.cfg.Config.context_length in
  let stop_token = fresh_token_at new_ids (simple_n - 2) in
  let stop_idx = ref 0 in
  while new_ids.(!stop_idx) <> stop_token do
    incr stop_idx
  done;
  Printf.printf "prompt ids: %s\n" (ids_str prompt);
  Printf.printf "reference: %d greedy tokens, eos=%d\n" n eos;
  Printf.printf "stop gate: token %d, first emitted at step %d\n" stop_token !stop_idx;

  let simple ~n ~eos_ids =
    Sampling.generate_simple_loop ~eos_ids ~temp:0.0 (Random.State.make [| 0 |])
      ~model_forward:(Gpt.forward model) ~context_length:ctx prompt ~max_gen:n
  in
  let cached ~n ~eos_ids =
    Gpt.generate_cached ~eos_ids ~temp:0.0 (Random.State.make [| 0 |]) model prompt ~max_gen:n
  in

  check_ids "gen_cached_eq_hf" ~prompt ~ours:(cached ~n ~eos_ids:None) ~ref_ids:new_ids;
  check_ids "gen_simple_eq_hf" ~prompt
    ~ours:(simple ~n:simple_n ~eos_ids:None)
    ~ref_ids:(Array.sub new_ids 0 simple_n);
  (* HF's stop contract, read off its own forced-EOS output: same length,
     same tokens, last one replaced by the stopping token. *)
  check "hf_eos_contract"
    (Array.length eos_ids = n && eos_ids.(n - 1) = eos && eos_ids.(n - 2) = new_ids.(n - 2));
  check_ids "stop_cached" ~prompt
    ~ours:(cached ~n ~eos_ids:(Some [ stop_token ]))
    ~ref_ids:(Array.sub new_ids 0 (!stop_idx + 1));
  check_ids "stop_simple" ~prompt
    ~ours:(simple ~n:simple_n ~eos_ids:(Some [ stop_token ]))
    ~ref_ids:(Array.sub new_ids 0 (!stop_idx + 1));

  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)