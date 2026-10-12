(** Model configs, mirroring [config.py].
    Only the GPT family is ported so far; Llama/Gemma/DeepSeek/Qwen
    configs follow the same record pattern. *)

type gpt_config = {
  vocab_size : int;
  context_length : int;
  emb_dim : int;
  n_heads : int;
  n_layers : int;
  drop_rate : float;
  qkv_bias : bool;
}

let gpt_config_124m =
  {
    vocab_size = 50257;
    context_length = 1024;
    emb_dim = 768;
    n_heads = 12;
    n_layers = 12;
    drop_rate = 0.0;
    qkv_bias = true;
  }

let gpt_small_config =
  {
    vocab_size = 50304;
    context_length = 256;
    emb_dim = 768;
    n_heads = 12;
    n_layers = 12;
    drop_rate = 0.0;
    qkv_bias = false;
  }

(** Mirrors [gpt2_config_creator] in [config.py]. *)
let gpt2_config_creator (size : string) : gpt_config =
  let base = gpt_config_124m in
  match size with
  | "gpt_s" -> { base with emb_dim = 768; n_layers = 12; n_heads = 12; qkv_bias = true }
  | "gpt_m" -> { base with emb_dim = 1024; n_layers = 24; n_heads = 16; qkv_bias = true }
  | "gpt_l" -> { base with emb_dim = 1280; n_layers = 36; n_heads = 20; qkv_bias = true }
  | "gpt_xl" -> { base with emb_dim = 1600; n_layers = 48; n_heads = 25; qkv_bias = true }
  | _ -> invalid_arg ("Config.gpt2_config_creator: unknown size " ^ size)

(** Tiny config for CPU-friendly tests and the demo binary. *)
let gpt_tiny_test_config =
  {
    vocab_size = 64;
    context_length = 16;
    emb_dim = 16;
    n_heads = 2;
    n_layers = 2;
    drop_rate = 0.0;
    qkv_bias = false;
  }
