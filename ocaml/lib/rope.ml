(** Rotary Position Embedding (RoPE), mirroring [llm_quest/common/rope.py]
     (the [RoPE] class — base form, without YaRN/llama3 scaling, which is
     step D4).

     RoPE encodes absolute position into query/key vectors by rotating
     each pair of feature dimensions by an angle proportional to the
     position. The defining property is that the inner product of a
     rotated query and a rotated key depends only on their *relative*
     position: [<R_m q, R_n k> = <q, R_{n-m} k>]. That is what makes it
     a relative-position scheme despite being applied as an absolute
     rotation.

     Conventions (matching the Python exactly, so parity is achievable):
     - Frequencies: [theta_i = base^(-2i / head_dim)] for [i = 0 .. head_dim/2 - 1].
     - Angles: [m * theta_i] for position [m]. The angle vector is
       **duplicated end-to-end** ([theta; theta]), *not* interleaved —
       so features [0 .. half-1] and [half .. head_dim-1] rotate by the
       same angle. This is the "halves" convention, not the paper's
       contiguous-pairs one.
     - Rotation: [rotate_half] maps [(x_0..x_{h-1}, x_h..x_{d-1})] to
       [(-x_h..-x_{d-1}, x_0..x_{h-1})].
     - Apply: [roped = cos * x + sin * rotate_half(x)].

     Shapes follow the per-head [mat] convention used throughout
     [Attention]: [x] is [(seq_len, head_dim)] and [cos]/[sin] are
     [(ctx_len, head_dim)] tables. [apply] selects rows from the tables:
     without [positions] it takes the first [seq_len] rows (prefill);
     with [positions] it gathers [cos.(p)] / [sin.(p)] per row (KV-cache
     decode, where the current token sits at an arbitrary position).

     Computed in OCaml's [float] (fp64) rather than the fp32 the Python
     enforces — strictly more precision, not a different function. *)

open Tensor

(** [compute_angles ~base ~head_dim ~ctx_len] returns [(cos, sin)], each
    of shape [(ctx_len, head_dim)]. [head_dim] must be even.

    [theta_i = base^(-2i/head_dim)], [angles[m][j] = m * theta_{j mod half}],
    and the angle row is duplicated so both halves share the same angles.
    [cos] and [sin] are the elementwise cosine/sine of those angles. *)
let compute_angles ~(base : float) ~(head_dim : int) ~(ctx_len : int) : mat * mat =
  if head_dim mod 2 <> 0 then invalid_arg "RoPE.compute_angles: head_dim must be even";
  let half = head_dim / 2 in
  let theta =
    Array.init half (fun i -> 1.0 /. (base ** (2.0 *. float_of_int i /. float_of_int head_dim)))
  in
  let row m =
    Array.init head_dim (fun j ->
        let angle = float_of_int m *. theta.(j mod half) in
        angle)
  in
  let cos = Array.init ctx_len (fun m -> Array.map cos (row m)) in
  let sin = Array.init ctx_len (fun m -> Array.map sin (row m)) in
  (cos, sin)

(** [rotate_half x] — split the last dim in half and swap with negation:
    [(x_0..x_{h-1}, x_h..x_{d-1})] -> [(-x_h..-x_{d-1}, x_0..x_{h-1})].
    Operates row-wise on a [(seq_len, head_dim)] matrix. *)
let rotate_half (x : mat) : mat =
  let hd = if Array.length x = 0 then 0 else Array.length x.(0) in
  let half = hd / 2 in
  Array.map
    (fun row ->
      Array.init hd (fun j ->
          if j < half then -.row.(j + half) else row.(j - half)))
    x

(** [apply ?positions ~cos ~sin x] — apply RoPE to [x (seq_len, head_dim)].

    Without [positions]: takes the first [seq_len] rows of [cos]/[sin]
    (the prefill case — positions are [0, 1, .., seq_len-1]).

    With [positions = Some ps]: gathers [cos.(ps.(i))] / [sin.(ps.(i))] for
    row [i] (the KV-cache decode case — the current token is at an
    arbitrary position [ps.(i)]). *)
let apply ?(positions : int array option = None) ~(cos : mat) ~(sin : mat) (x : mat) : mat =
  let s = Array.length x in
  let hd = if s = 0 then 0 else Array.length x.(0) in
  let cos_i, sin_i =
    match positions with
    | None ->
        if Array.length cos < s || Array.length sin < s then
          invalid_arg "RoPE.apply: cos/sin table shorter than seq_len";
        ((fun i -> cos.(i)), (fun i -> sin.(i)))
    | Some ps ->
        if Array.length ps <> s then invalid_arg "RoPE.apply: positions length mismatch";
        ((fun i -> cos.(ps.(i))), (fun i -> sin.(ps.(i))))
  in
  let rot = rotate_half x in
  Array.init s (fun i ->
      let c = cos_i i in
      let sn = sin_i i in
      Array.init hd (fun j -> c.(j) *. x.(i).(j) +. sn.(j) *. rot.(i).(j)))

(** [compute_angles_llama3 ~base ~head_dim ~ctx_len ~scaling] — llama3 RoPE
    scaling, mirroring [_compute_llama3_parameters] in transformers'
    [modeling_rope_utils.py].

    The scaling adjusts inverse frequencies by wavelength:
    - wavelen > low_freq_wavelen: divide by factor (full interpolation)
    - wavelen < high_freq_wavelen: no change (high freq, in-window)
    - otherwise: smooth interpolation between the two

    [scaling] is the [Config.rope_scaling] [Llama3] variant, carrying
    [factor], [low_freq_factor], [high_freq_factor], and
    [original_max_position_embeddings]. *)
let compute_angles_llama3 ~(base : float) ~(head_dim : int) ~(ctx_len : int)
    ~(scaling : Config.rope_scaling) : mat * mat =
  match scaling with
  | Llama3 { factor; low_freq_factor; high_freq_factor; original_max_position_embeddings } ->
      let half = head_dim / 2 in
      let old_context_len = float_of_int original_max_position_embeddings in
      let low_freq_wavelen = old_context_len /. low_freq_factor in
      let high_freq_wavelen = old_context_len /. high_freq_factor in

      (* Compute scaled inverse frequencies *)
      let inv_freq =
        Array.init half (fun i ->
            let exponent = 2.0 *. float_of_int i /. float_of_int head_dim in
            let base_inv = 1.0 /. (base ** exponent) in
            let wavelen = 2.0 *. Float.pi /. base_inv in
            if wavelen > low_freq_wavelen then
              base_inv /. factor
            else if wavelen < high_freq_wavelen then
              base_inv
            else begin
              let smooth_factor =
                (old_context_len /. wavelen -. low_freq_factor)
                /. (high_freq_factor -. low_freq_factor)
              in
              let inv_freq_llama = base_inv in
              ((1.0 -. smooth_factor) *. inv_freq_llama /. factor
               +. smooth_factor *. inv_freq_llama)
            end)
      in

      (* Build cos/sin tables from scaled frequencies *)
      let row m =
        Array.init head_dim (fun j ->
            let angle = float_of_int m *. inv_freq.(j mod half) in
            angle)
      in
      let cos = Array.init ctx_len (fun m -> Array.map cos (row m)) in
      let sin = Array.init ctx_len (fun m -> Array.map sin (row m)) in
      (cos, sin)
