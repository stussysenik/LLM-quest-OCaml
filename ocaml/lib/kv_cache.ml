(** Dynamic KV cache, mirroring [KVCache] in [llm_quest/utils.py].

    The Python version pre-allocates [num_layers] tensors of shape
    [(b, heads, capacity, head_dim)] and grows them by chunks.
    This port keeps the same externally visible behaviour with plain
    OCaml arrays (single batch, for clarity):

    - one slot per layer holding all past per-head K/V rows;
    - [push] appends the current step's K/V (usually 1 row / head);
    - [peek] returns the prefix that [Attention.mha_forward]'s [~kv]
      argument prepends to the current K/V.

    Invariant: [peek l] returns keys/values for positions
    [[0, seq_len_l)], then the caller forwards the new token(s) and
    [push]es their K/V. Positions therefore line up with the causal
    mask logic in [Attention.head_forward]. *)

open Tensor

type layer_kv = { keys : mat array; values : mat array }
(** Per-layer cache: one [(seq_past x head_dim)] matrix per head. *)

type t = { num_layers : int; mutable layers : layer_kv option array }

let create ~num_layers : t = { num_layers; layers = Array.make num_layers None }

let peek (c : t) (layer_idx : int) : layer_kv option = c.layers.(layer_idx)

let past_len (c : t) (layer_idx : int) : int =
  match c.layers.(layer_idx) with
  | None -> 0
  | Some kv ->
      if Array.length kv.keys = 0 then 0 else Array.length kv.keys.(0)

let push (c : t) ~layer_idx ~(keys : mat array) ~(values : mat array) : unit =
  let h = Array.length keys in
  (match c.layers.(layer_idx) with
  | None -> c.layers.(layer_idx) <- Some { keys = Array.map copy_mat keys; values = Array.map copy_mat values }
  | Some kv ->
      if Array.length kv.keys <> h then invalid_arg "Kv_cache.push: head count changed";
      for i = 0 to h - 1 do
        kv.keys.(i) <- Array.append kv.keys.(i) keys.(i);
        kv.values.(i) <- Array.append kv.values.(i) values.(i)
      done)
