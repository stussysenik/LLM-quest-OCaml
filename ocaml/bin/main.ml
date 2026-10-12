(** Demo: tiny GPT-2 forward + greedy generation, all in OCaml.
    Mirrors PART A of the `__main__` demo in [llm_quest/generate.py]
    (generation without pretrained weights). *)

let () =
  let rs = Random.State.make [| 123 |] in
  let cfg = Config.gpt_tiny_test_config in
  Printf.printf "cfg: vocab=%d ctx=%d emb=%d heads=%d layers=%d\n" cfg.Config.vocab_size
    cfg.Config.context_length cfg.Config.emb_dim cfg.Config.n_heads cfg.Config.n_layers;
  let model = Gpt.make rs cfg () in
  Printf.printf "params: %d\n" (Gpt.num_parameters model);
  let prompt = [| 1; 2; 3 |] in
  let logits = Gpt.forward model prompt in
  Printf.printf "logits shape: (%d, %d)\n" (Array.length logits) (Array.length logits.(0));
  Printf.printf "first-row logits[0..7]:";
  for j = 0 to 7 do
    Printf.printf " %.4f" logits.(0).(j)
  done;
  print_newline ();
  let out =
    Sampling.generate_simple_loop ~temp:0.0 rs ~model_forward:(Gpt.forward model)
      ~context_length:cfg.Config.context_length prompt ~max_gen:8
  in
  Printf.printf "prompt: [%s]\n"
    (String.concat ";" (List.map string_of_int (Array.to_list prompt)));
  Printf.printf "generated: [%s]\n"
    (String.concat ";" (List.map string_of_int (Array.to_list out)))
