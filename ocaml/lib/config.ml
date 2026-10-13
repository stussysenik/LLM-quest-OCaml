(** Model configs, mirroring [config.py] (GPT family) and Llama 3.

    One record carries both families. The fields every decoder-only
    transformer has live at the top level; everything that actually
    differs sits behind [arch]. Callers go through [max_positions],
    [norm_eps], [ffn_inner], [kv_heads], [head_dim] — accessors that
    push the variant match to the edges, so shared code (block stack,
    attention, cache, loader) never branches on family. Llama-only
    knobs ([rope_theta], [rope_scaling], [tie_word_embeddings]) raise on
    a GPT-2 config rather than returning a plausible-looking default: a
    wrong rope_theta is a silent numerical failure, and an explicit
    raise is the cheapest way to find it.

    [qkv_bias] is an accessor rather than a variant field because it is
    a real architectural difference with no shared meaning: GPT-2's
    Conv1D projections carry bias, Llama's linears do not. *)

type rope_scaling =
  | Llama3 of {
      factor : float;
      low_freq_factor : float;
      high_freq_factor : float;
      original_max_position_embeddings : int;
    }

(** [Gelu] is the exact-erf GELU ([gpt_transformer_block.py]),
    [Gelu_new] HuggingFace's tanh approximation ("gelu_new", the GPT-2
    checkpoint default). Both families name only activations the [Norm]
    module can actually compute — add a case there first, not here. *)
type activation = [ `Gelu | `Gelu_new ]

type arch =
  | Gpt2 of {
      n_positions : int;
      n_inner : int option; (** None = 4 * hidden_size *)
      layer_norm_eps : float;
      activation : activation;
      qkv_bias : bool;
    }
  | Llama of {
      num_key_value_heads : int;
      intermediate_size : int;
      max_position_embeddings : int;
      rms_norm_eps : float;
      rope_theta : float;
      rope_scaling : rope_scaling option;
      tie_word_embeddings : bool;
    }

type t = {
  vocab_size : int;
  hidden_size : int;
  num_layers : int;
  num_heads : int;
  arch : arch;
}

let head_dim (c : t) : int = c.hidden_size / c.num_heads

let max_positions (c : t) : int =
  match c.arch with
  | Gpt2 g -> g.n_positions
  | Llama l -> l.max_position_embeddings

let norm_eps (c : t) : float =
  match c.arch with
  | Gpt2 g -> g.layer_norm_eps
  | Llama l -> l.rms_norm_eps

let ffn_inner (c : t) : int =
  match c.arch with
  | Gpt2 g -> ( match g.n_inner with Some n -> n | None -> 4 * c.hidden_size)
  | Llama l -> l.intermediate_size

(** K/V heads per layer. GPT-2 is full MHA (one K/V head per Q head);
    Llama is GQA, where each K/V head serves [num_heads / kv_heads]
    queries — the KV cache is sized by this, not by [num_heads]. *)
let kv_heads (c : t) : int =
  match c.arch with
  | Gpt2 _ -> c.num_heads
  | Llama l -> l.num_key_value_heads

let qkv_bias (c : t) : bool =
  match c.arch with
  | Gpt2 g -> g.qkv_bias
  | Llama _ -> false

(** GPT-2 has a learned [wpe] table; Llama derives positions from RoPE
    and has no such tensor. *)
let has_learned_positions (c : t) : bool =
  match c.arch with
  | Gpt2 _ -> true
  | Llama _ -> false

let activation (c : t) : activation =
  match c.arch with
  | Gpt2 g -> g.activation
  | Llama _ -> invalid_arg "Config.activation: not a GPT-2 config (Llama uses SwiGLU)"

let is_llama (c : t) : bool = match c.arch with Gpt2 _ -> false | Llama _ -> true

let rope_theta (c : t) : float =
  match c.arch with
  | Llama l -> l.rope_theta
  | Gpt2 _ -> invalid_arg "Config.rope_theta: not a Llama config"

let rope_scaling (c : t) : rope_scaling option =
  match c.arch with
  | Llama l -> l.rope_scaling
  | Gpt2 _ -> invalid_arg "Config.rope_scaling: not a Llama config"

let tie_word_embeddings (c : t) : bool =
  match c.arch with
  | Llama l -> l.tie_word_embeddings
  | Gpt2 _ -> invalid_arg "Config.tie_word_embeddings: not a Llama config"

(* ------------------------------------------------------------------ *)
(* GPT family (config.py)                                              *)
(* ------------------------------------------------------------------ *)

let gpt_config_124m =
  {
    vocab_size = 50257;
    hidden_size = 768;
    num_layers = 12;
    num_heads = 12;
    arch =
      Gpt2
        {
          n_positions = 1024;
          n_inner = None;
          layer_norm_eps = 1e-5;
          (* The HF `gpt2` checkpoint uses `gelu_new`; the loader used to
             hardcode this, which was a second source of truth. *)
          activation = `Gelu_new;
          qkv_bias = true;
        };
  }

let gpt_small_config =
  {
    vocab_size = 50304;
    hidden_size = 768;
    num_layers = 12;
    num_heads = 12;
    arch =
      Gpt2
        {
          n_positions = 256;
          n_inner = None;
          layer_norm_eps = 1e-5;
          activation = `Gelu_new;
          qkv_bias = false;
        };
  }

(** Mirrors [gpt2_config_creator] in [config.py]. *)
let gpt2_config_creator (size : string) : t =
  match size with
  | "gpt_s" -> gpt_config_124m
  | "gpt_m" -> { gpt_config_124m with hidden_size = 1024; num_layers = 24; num_heads = 16 }
  | "gpt_l" -> { gpt_config_124m with hidden_size = 1280; num_layers = 36; num_heads = 20 }
  | "gpt_xl" -> { gpt_config_124m with hidden_size = 1600; num_layers = 48; num_heads = 25 }
  | _ -> invalid_arg ("Config.gpt2_config_creator: unknown size " ^ size)

(** Tiny GPT-2 for CPU-friendly tests and the demo binary. [Gelu] (exact)
    is what the self-consistency gates were written against. *)
let gpt_tiny_test_config =
  {
    vocab_size = 64;
    hidden_size = 16;
    num_layers = 2;
    num_heads = 2;
    arch =
      Gpt2
        {
          n_positions = 16;
          n_inner = None;
          layer_norm_eps = 1e-5;
          activation = `Gelu;
          qkv_bias = false;
        };
  }

(* ------------------------------------------------------------------ *)
(* Llama family                                                         *)
(* ------------------------------------------------------------------ *)

(** Tiny Llama for the architecture gates (RMSNorm/SwiGLU/RoPE/GQA) and
    the tiny-random parity run. GQA on purpose: [kv_heads < num_heads]
    is the configuration that catches head-grouping bugs, and a tiny
    config with no GQA would not exercise them. *)
let llama_tiny_test_config =
  {
    vocab_size = 64;
    hidden_size = 64;
    num_layers = 2;
    num_heads = 4;
    arch =
      Llama
        {
          num_key_value_heads = 2;
          intermediate_size = 128;
          max_position_embeddings = 16;
          rms_norm_eps = 1e-5;
          rope_theta = 10000.0;
          rope_scaling = None;
          tie_word_embeddings = true;
        };
  }