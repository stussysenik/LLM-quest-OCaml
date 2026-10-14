(** Minimal safetensors reader (stdlib only).

    Format: [[8 bytes: u64 LE header length][JSON header][raw data]].
    The header maps names to [{dtype, shape, data_offsets}]; data is
    contiguous little-endian raw tensors in header order.

    Only [F32] and [I64] dtypes are supported (everything `gpt2`
    needs). Tensors are decoded lazily, one at a time, straight into
    the destination [float array]s — the 550MB `gpt2` file is held
    once as [bytes], never duplicated as intermediate arrays. *)

type dtype = F32 | F64 | I64

type entry = { dtype : dtype; shape : int list; off_start : int; off_end : int }

type file = { data : bytes; entries : (string * entry) list }

(* ---------- tiny JSON parser (objects/arrays/strings/ints only) ---------- *)

type json = JObj of (string * json) list | JArr of json list | JStr of string | JNum of int

let parse_json (s : string) : json =
  let n = String.length s in
  let pos = ref 0 in
  let skip () =
    while !pos < n do
      match s.[!pos] with ' ' | '\t' | '\n' | '\r' -> incr pos | _ -> raise Exit
    done
  in
  let skip_ws () = try skip () with Exit -> () in
  let expect c =
    skip_ws ();
    if !pos >= n || s.[!pos] <> c then failwith "Safetensors: malformed header";
    incr pos
  in
  let rec parse_value () =
    skip_ws ();
    if !pos >= n then failwith "Safetensors: truncated header";
    (match s.[!pos] with
    | '{' -> parse_obj ()
    | '[' -> parse_arr ()
    | '"' -> JStr (parse_string ())
    | c when c = '-' || (c >= '0' && c <= '9') -> JNum (parse_num ())
    | _ -> failwith "Safetensors: unexpected char in header")
  and parse_obj () =
    expect '{';
    skip_ws ();
    if !pos < n && s.[!pos] = '}' then (
      incr pos;
      JObj [])
    else
      let fields = ref [] in
      let more = ref true in
      while !more do
        skip_ws ();
        let k = parse_string () in
        expect ':';
        let v = parse_value () in
        fields := (k, v) :: !fields;
        skip_ws ();
        if !pos < n && s.[!pos] = ',' then incr pos else more := false
      done;
      expect '}';
      JObj (List.rev !fields)
  and parse_arr () =
    expect '[';
    skip_ws ();
    if !pos < n && s.[!pos] = ']' then (
      incr pos;
      JArr [])
    else
      let items = ref [] in
      let more = ref true in
      while !more do
        items := parse_value () :: !items;
        skip_ws ();
        if !pos < n && s.[!pos] = ',' then incr pos else more := false
      done;
      expect ']';
      JArr (List.rev !items)
  and parse_string () =
    expect '"';
    let buf = Buffer.create 32 in
    let closed = ref false in
    while not !closed do
      if !pos >= n then failwith "Safetensors: unterminated string";
      let c = s.[!pos] in
      incr pos;
      if c = '"' then closed := true
      else if c = '\\' then (
        if !pos >= n then failwith "Safetensors: bad escape";
        let e = s.[!pos] in
        incr pos;
        Buffer.add_char buf
          (match e with
          | '"' -> '"'
          | '\\' -> '\\'
          | '/' -> '/'
          | 'n' -> '\n'
          | 't' -> '\t'
          | 'r' -> '\r'
          | _ -> failwith "Safetensors: bad escape"))
      else Buffer.add_char buf c
    done;
    Buffer.contents buf
  and parse_num () =
    let start = !pos in
    if !pos < n && s.[!pos] = '-' then incr pos;
    while !pos < n && s.[!pos] >= '0' && s.[!pos] <= '9' do
      incr pos
    done;
    int_of_string (String.sub s start (!pos - start))
  in
  skip_ws ();
  let v = parse_value () in
  skip_ws ();
  if !pos <> n then failwith "Safetensors: trailing bytes in header";
  v

(* ---------- file loading ---------- *)

let open_file (path : string) : file =
  let ic = open_in_bin path in
  let len = in_channel_length ic in
  let data = Bytes.create len in
  really_input ic data 0 len;
  close_in ic;
  if len < 8 then failwith "Safetensors: file too short";
  let hlen = ref 0L in
  for k = 0 to 7 do
    hlen := Int64.logor !hlen (Int64.shift_left (Int64.of_int (Bytes.get_uint8 data k)) (8 * k))
  done;
  let hlen = Int64.to_int !hlen in
  if 8 + hlen > len then failwith "Safetensors: header overruns file";
  let hstr = Bytes.sub_string data 8 hlen in
  let base = 8 + hlen in
  match parse_json hstr with
  | JObj fields ->
      let entries =
        List.filter_map
          (fun (name, v) ->
            if name = "__metadata__" then None
            else
              match v with
              | JObj f ->
                  let dtype =
                    match List.assoc "dtype" f with
                    | JStr "F32" -> F32
                    | JStr "F64" -> F64
                    | JStr "I64" -> I64
                    | JStr d -> failwith ("Safetensors: unsupported dtype " ^ d)
                    | _ -> failwith "Safetensors: bad dtype field"
                  in
                  let shape =
                    match List.assoc "shape" f with
                    | JArr xs ->
                        List.map
                          (function JNum i -> i | _ -> failwith "Safetensors: bad shape")
                          xs
                    | _ -> failwith "Safetensors: bad shape field"
                  in
                  let a, b =
                    match List.assoc "data_offsets" f with
                    | JArr [ JNum a; JNum b ] -> (a, b)
                    | _ -> failwith "Safetensors: bad data_offsets"
                  in
                  if base + b > len then failwith ("Safetensors: offsets overrun file: " ^ name);
                  Some (name, { dtype; shape; off_start = base + a; off_end = base + b })
              | _ -> failwith "Safetensors: bad tensor entry")
          fields
      in
      { data; entries }
  | _ -> failwith "Safetensors: header must be an object"

(* ---------- typed accessors ---------- *)

let find (f : file) (name : string) : entry =
  try List.assoc name f.entries with Not_found -> failwith ("Safetensors: tensor not found: " ^ name)

let names (f : file) : string list = List.map fst f.entries

let shape_str (s : int list) : string = "[" ^ String.concat ";" (List.map string_of_int s) ^ "]"

(** A tensor this loader must NOT find. Llama has no biases anywhere and
    no learned positions; a GPT-2-shaped loader that simply stops asking
    for [*.bias] reads a Llama file "successfully" and leaves every
    projection un-biased by luck rather than by check. *)
let require_absent (f : file) (name : string) : unit =
  if List.mem_assoc name f.entries then
    failwith (Printf.sprintf "Safetensors: unexpected tensor %s" name)

(** Every tensor in the file must be claimed, and every claimed tensor
    must exist. [skip_prefix] covers the [ref.*] namespace: the parity
    harness keeps weights and reference activations in one file, and the
    reference is not the loader's business. Everything else must match
    [expected] exactly — so a GPT-2 file handed to the Llama loader
    fails here naming the first unexpected key, which is the right way
    to find out. *)
let assert_exact_keys (f : file) ~(skip_prefix : string) (expected : string list) : unit =
  let claimed = List.sort_uniq compare expected in
  let present =
    List.sort_uniq compare (List.filter (fun k -> not (String.starts_with ~prefix:skip_prefix k)) (names f))
  in
  let missing = List.filter (fun k -> not (List.mem k present)) claimed in
  let unexpected = List.filter (fun k -> not (List.mem k claimed)) present in
  match (missing, unexpected) with
  | [], [] -> ()
  | _ ->
      failwith
        (Printf.sprintf "Safetensors: file does not match the loader's key set — missing: [%s], unexpected: [%s]"
           (String.concat "; " missing)
           (String.concat "; " unexpected))

let mat_of (f : file) (name : string) : Tensor.mat =
  let e = find f name in
  if e.dtype <> F32 then failwith ("Safetensors.mat_of: not F32: " ^ name);
  let rows, cols =
    match e.shape with
    | [ r; c ] -> (r, c)
    | _ -> failwith ("Safetensors.mat_of: not 2D: " ^ name)
  in
  if e.off_end - e.off_start <> rows * cols * 4 then
    failwith ("Safetensors.mat_of: size mismatch: " ^ name);
  Array.init rows (fun i ->
      Array.init cols (fun j ->
          let o = e.off_start + ((i * cols) + j) * 4 in
          Int32.float_of_bits (Bytes.get_int32_le f.data o)))

(** One head slice of a 3D [(heads, seq, dim)] tensor (e.g. exported
    `past_key_values`: head [h] owns contiguous rows [h*seq, (h+1)*seq)
    in row-major order). *)
let mat_of_head (f : file) (name : string) ~(head : int) ~(heads : int) : Tensor.mat =
  let e = find f name in
  if e.dtype <> F32 then failwith ("Safetensors.mat_of_head: not F32: " ^ name);
  let h, s, d =
    match e.shape with
    | [ h; s; d ] -> (h, s, d)
    | _ -> failwith ("Safetensors.mat_of_head: not 3D: " ^ name)
  in
  if h <> heads || head < 0 || head >= h then failwith ("Safetensors.mat_of_head: bad head: " ^ name);
  if e.off_end - e.off_start <> h * s * d * 4 then
    failwith ("Safetensors.mat_of_head: size mismatch: " ^ name);
  Array.init s (fun i ->
      Array.init d (fun j ->
          let o = e.off_start + (((head * s) + i) * d + j) * 4 in
          Int32.float_of_bits (Bytes.get_int32_le f.data o)))

(** Read a 2D [F64] tensor as a [Tensor.mat] of OCaml [float]s.

    **Why reference tensors are F64 and weights stay F32.** A parity gate
    compares two implementations of the same function; the tolerance it
    can hold is set by the *less precise* of the two. HF runs in fp32, and
    fp32 is not accurate to 1e-6 relative once a 12-layer block stack has
    amplified the ~3000-magnitude GPT-2 outlier dims: on this machine
    PyTorch disagrees with **its own fp64 run** by 2.3e-6 relative at
    block 3. Gating a faithful fp64 OCaml port against an fp32 reference
    at 1e-6 therefore measures the reference's own rounding, not our
    error — and it is platform-dependent besides, because a different CPU
    BLAS reduces in a different order.

    So the *reference activations* are exported in float64 while the
    *weights* stay float32, which is what they are on disk. Both sides
    then accumulate in fp64 from bit-identical inputs, and what remains is
    summation order — around 1e-14 relative, three orders of magnitude
    tighter than the old gate and, unlike it, the same number on any
    machine. Verified, not assumed: this is exactly the comparison that
    passes at 1e-9 and would not have passed at 1e-6 before. *)
let mat_of_f64 (f : file) (name : string) : Tensor.mat =
  let e = find f name in
  if e.dtype <> F64 then failwith ("Safetensors.mat_of_f64: not F64: " ^ name);
  let rows, cols =
    match e.shape with
    | [ r; c ] -> (r, c)
    | _ -> failwith ("Safetensors.mat_of_f64: not 2D: " ^ name)
  in
  if e.off_end - e.off_start <> rows * cols * 8 then
    failwith ("Safetensors.mat_of_f64: size mismatch: " ^ name);
  Array.init rows (fun i ->
      Array.init cols (fun j ->
          let o = e.off_start + ((i * cols) + j) * 8 in
          Int64.float_of_bits (Bytes.get_int64_le f.data o)))

(** 3D [F64] tensor as nested OCaml arrays — an image is
    [num_channels, height, width], which is one level more than a [mat]
    and cannot be squeezed into it without a reshape whose correctness is
    then nobody's problem. The ViT references store images this way. *)
let mat3_of_f64 (f : file) (name : string) : float array array array =
  let e = find f name in
  if e.dtype <> F64 then failwith ("Safetensors.mat3_of_f64: not F64: " ^ name);
  let d0, d1, d2 =
    match e.shape with
    | [ a; b; c ] -> (a, b, c)
    | _ -> failwith ("Safetensors.mat3_of_f64: not 3D: " ^ name)
  in
  if e.off_end - e.off_start <> d0 * d1 * d2 * 8 then
    failwith ("Safetensors.mat3_of_f64: size mismatch: " ^ name);
  let get i j k =
    Int64.float_of_bits (Bytes.get_int64_le f.data (e.off_start + (((i * d1) + j) * d2 + k) * 8))
  in
  Array.init d0 (fun i -> Array.init d1 (fun j -> Array.init d2 (fun k -> get i j k)))

(** 1D [F64] twin of [mat_of_f64] — the norm weights, biases and CLS
    tokens in the ViT/VLM references are all 1D. *)
let vec_of_f64 (f : file) (name : string) : Tensor.vec =
  let e = find f name in
  if e.dtype <> F64 then failwith ("Safetensors.vec_of_f64: not F64: " ^ name);
  let n =
    match e.shape with [ n ] -> n | _ -> failwith ("Safetensors.vec_of_f64: not 1D: " ^ name)
  in
  if e.off_end - e.off_start <> n * 8 then failwith ("Safetensors.vec_of_f64: size mismatch: " ^ name);
  Array.init n (fun i ->
      Int64.float_of_bits (Bytes.get_int64_le f.data (e.off_start + (i * 8))))

let vec_of (f : file) (name : string) : Tensor.vec =
  let e = find f name in
  if e.dtype <> F32 then failwith ("Safetensors.vec_of: not F32: " ^ name);
  let n =
    match e.shape with [ n ] -> n | _ -> failwith ("Safetensors.vec_of: not 1D: " ^ name)
  in
  if e.off_end - e.off_start <> n * 4 then failwith ("Safetensors.vec_of: size mismatch: " ^ name);
  Array.init n (fun i ->
      Int32.float_of_bits (Bytes.get_int32_le f.data (e.off_start + (i * 4))))

(** Read a tensor, asserting the shape it *must* have.

    Every loader goes through this, and the expectation always comes
    from [Config] — never from the file. That is the whole point: a
    transposed `nn.Linear` weight has a square shape for
    [o_proj]/[q_proj] and loads silently, producing wrong numbers with
    no error anywhere; only an expected shape catches it. The message
    names the tensor, so a config/file mismatch is identifiable without
    a debugger. *)
let get_tensor (f : file) (name : string) ~(expected_shape : int list) : Tensor.mat =
  let e = find f name in
  if e.shape <> expected_shape then
    failwith
      (Printf.sprintf "Safetensors: %s has shape %s, expected %s" name (shape_str e.shape)
         (shape_str expected_shape));
  mat_of f name

(** 1D twin of [get_tensor] — Llama's norms are bare weight vectors with
    no bias, and GPT-2's LayerNorms are scale + shift. *)
let get_vector (f : file) (name : string) ~(expected_len : int) : Tensor.vec =
  let e = find f name in
  if e.shape <> [ expected_len ] then
    failwith
      (Printf.sprintf "Safetensors: %s has shape %s, expected [%d]" name (shape_str e.shape)
         expected_len);
  vec_of f name

let i64_array_of (f : file) (name : string) : int64 array =
  let e = find f name in
  if e.dtype <> I64 then failwith ("Safetensors.i64_array_of: not I64: " ^ name);
  let n =
    match e.shape with [ n ] -> n | _ -> failwith ("Safetensors.i64_array_of: not 1D: " ^ name)
  in
  if e.off_end - e.off_start <> n * 8 then
    failwith ("Safetensors.i64_array_of: size mismatch: " ^ name);
  Array.init n (fun i -> Bytes.get_int64_le f.data (e.off_start + (i * 8)))
