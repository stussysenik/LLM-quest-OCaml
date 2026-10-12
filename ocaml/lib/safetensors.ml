(** Minimal safetensors reader (stdlib only).

    Format: [[8 bytes: u64 LE header length][JSON header][raw data]].
    The header maps names to [{dtype, shape, data_offsets}]; data is
    contiguous little-endian raw tensors in header order.

    Only [F32] and [I64] dtypes are supported (everything `gpt2`
    needs). Tensors are decoded lazily, one at a time, straight into
    the destination [float array]s — the 550MB `gpt2` file is held
    once as [bytes], never duplicated as intermediate arrays. *)

type dtype = F32 | I64

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

(** Row-major transpose on decode (for `nn.Linear`-style `(out, in)` weights). *)
let mat_of_transposed (f : file) (name : string) : Tensor.mat = Tensor.transpose (mat_of f name)

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

let vec_of (f : file) (name : string) : Tensor.vec =
  let e = find f name in
  if e.dtype <> F32 then failwith ("Safetensors.vec_of: not F32: " ^ name);
  let n =
    match e.shape with [ n ] -> n | _ -> failwith ("Safetensors.vec_of: not 1D: " ^ name)
  in
  if e.off_end - e.off_start <> n * 4 then failwith ("Safetensors.vec_of: size mismatch: " ^ name);
  Array.init n (fun i ->
      Int32.float_of_bits (Bytes.get_int32_le f.data (e.off_start + (i * 4))))

let i64_array_of (f : file) (name : string) : int64 array =
  let e = find f name in
  if e.dtype <> I64 then failwith ("Safetensors.i64_array_of: not I64: " ^ name);
  let n =
    match e.shape with [ n ] -> n | _ -> failwith ("Safetensors.i64_array_of: not 1D: " ^ name)
  in
  if e.off_end - e.off_start <> n * 8 then
    failwith ("Safetensors.i64_array_of: size mismatch: " ^ name);
  Array.init n (fun i -> Bytes.get_int64_le f.data (e.off_start + (i * 8)))
