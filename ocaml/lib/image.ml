(** Image preprocessing for the multimodal path.

    [vlm_generation.py] takes a PIL image and runs three steps before the
    ViT sees it:

    {1 F.resize([S, S])}          bilinear, antialiased, on **uint8**
    {2 F.to_tensor}               uint8 -> float, divided by 255
    {3 F.normalize}               ImageNet mean/std

    **What is and is not claimed here.** Steps 2 and 3 are pure
    arithmetic — a divide and a per-channel affine — so they are
    reproduced exactly and gated at 1e-9 against PyTorch. Step 1 is
    *not* reproduced: torchvision's default [F.resize] antialiases, and
    an antialiased resampler is a filter-bank specification, not a
    formula. [resize_bilinear] below is [torch.nn.functional.interpolate]'s
    [mode="bilinear", align_corners=False] — the same sample positions,
    without the antialias filter and without the uint8 rounding
    torchvision applies because it resamples the PIL image in the integer
    domain. The reference this is gated against is therefore torch's
    resampler, *not* [torchvision.transforms.functional.resize], and the
    test says so in its own output. Anyone who needs bit-identical
    captions from a real checkpoint needs a real resampler; what this
    module buys is that the rest of the pipeline — the tensor layout, the
    scaling, the normalisation constants — is exact and gated, so a
    divergence downstream has one known cause instead of two.

    The layout throughout is [Vit.image] = [C][H][W] of floats, which is
    the type the ViT's patch embedding already consumes. Converting is
    [to_tensor]'s job; nothing else in the port sees channel-last. *)

open Tensor

(** ImageNet normalisation constants, the ones
    [torchvision.transforms.Normalize] defaults to and the ones
    [MultimodalDataset] uses. Per-channel, ordered RGB. *)
let imagenet_mean = [| 0.485; 0.456; 0.406 |]

let imagenet_std = [| 0.229; 0.224; 0.225 |]

(** Raw pixels as they come out of a decoder: [H][W][C] ints in
    [0, 255]. Channel-last, because that is how images are stored
    everywhere outside this port; the transpose into [C][H][W] happens
    exactly once, in [to_tensor]. *)
type raw = int array array array

(** Step 2, and the layout change: uint8 [H][W][C] -> float [C][H][W]
    scaled by 1/255.

    The division is by 255 rather than 256 because [F.to_tensor] divides
    by 255, and the off-by-one matters: at a white pixel the two differ
    by 1/65280, which is 1.5e-5 — three orders above every gate here. *)
let to_tensor (px : raw) : Vit.image =
  let h = Array.length px in
  if h = 0 then invalid_arg "Image.to_tensor: empty image";
  let w = Array.length px.(0) in
  let c = Array.length px.(0).(0) in
  if c <> 3 then
    invalid_arg (Printf.sprintf "Image.to_tensor: expected 3 channels (RGB), got %d" c);
  Array.init c (fun ci ->
      Array.init h (fun y ->
          Array.init w (fun x ->
              let v = px.(y).(x).(ci) in
              if v < 0 || v > 255 then
                invalid_arg (Printf.sprintf "Image.to_tensor: pixel %d out of range [0,255]" v);
              float_of_int v /. 255.0)))

(** Step 3: per-channel [(x - mean) / std]. In place of a fused affine
    because the mean and std differ per channel and fusing them would
    hide which constant is which. *)
let normalize_imagenet (x : Vit.image) : Vit.image =
  let c = Array.length x in
  if c <> Array.length imagenet_mean then
    invalid_arg
      (Printf.sprintf "Image.normalize_imagenet: %d channels, ImageNet constants are %d" c
         (Array.length imagenet_mean));
  Array.mapi
    (fun ci plane ->
      let mu = imagenet_mean.(ci) and sd = imagenet_std.(ci) in
      Array.map
        (fun row -> Array.map (fun v -> (v -. mu) /. sd) row)
        plane)
    x

let normalize_image ~(mean : float array) ~(std : float array) (x : Vit.image) : Vit.image =
  if Array.length x <> Array.length mean || Array.length mean <> Array.length std then
    invalid_arg "Image.normalize_image: channel count must match mean and std";
  Array.mapi
    (fun ci plane ->
      let mu = mean.(ci) and sd = std.(ci) in
      if sd = 0.0 then invalid_arg "Image.normalize_image: zero std";
      Array.map (fun row -> Array.map (fun v -> (v -. mu) /. sd) row) plane)
    x

(** Step 1, the honest version: bilinear resampling at
    [torch.nn.functional.interpolate]'s sample positions.

    [align_corners=False] maps output index [d] to source coordinate
    [(d + 0.5) * in / out - 0.5], clamps negatives to 0, and interpolates
    between the two neighbouring rows/cols. That formula is reproduced
    exactly; the antialias filter and the uint8 rounding that
    [torchvision] layers on top are not, and the header says so.

    Half-pixel centres matter more than they look: using
    [d * in / out] instead shifts every sample by half a source pixel,
    which on a 224->256 upscale is a visible 2-pixel error and on a
    downscale is a blur — neither of which any shape assertion catches. *)
let resize_bilinear ~size (x : Vit.image) : Vit.image =
  if size <= 0 then invalid_arg "Image.resize_bilinear: size must be positive";
  let c = Array.length x in
  if c = 0 then invalid_arg "Image.resize_bilinear: image has no channels";
  let h_in = Array.length x.(0) in
  if h_in = 0 then invalid_arg "Image.resize_bilinear: image has no rows";
  let w_in = Array.length x.(0).(0) in
  if w_in = 0 then invalid_arg "Image.resize_bilinear: image has no columns";
  (* One axis: output index -> (lower source index, upper source index,
     weight of the upper one). Written once because rows and columns use
     the identical rule with different lengths, and having it spelled out
     twice is how a half-pixel centre ends up in one axis and not the
     other. *)
  let axis src_max dst_max i =
    let scale = float_of_int src_max /. float_of_int dst_max in
    let src = ((float_of_int i +. 0.5) *. scale) -. 0.5 in
    let src = if src < 0.0 then 0.0 else src in
    let last = src_max - 1 in
    let i0 = int_of_float (Float.floor src) in
    let i0 = if i0 > last then last else i0 in
    let i1 = if i0 + 1 < src_max then i0 + 1 else last in
    (i0, i1, src -. float_of_int i0)
  in
  let sy = axis h_in size and sx = axis w_in size in
  Array.init c (fun ci ->
      let plane = x.(ci) in
      Array.init size (fun y ->
          let (y0, y1, wy) = sy y in
          Array.init size (fun xx ->
              let (x0, x1, wx) = sx xx in
              let top = (plane.(y0).(x0) *. (1.0 -. wx)) +. (plane.(y0).(x1) *. wx) in
              let bot = (plane.(y1).(x0) *. (1.0 -. wx)) +. (plane.(y1).(x1) *. wx) in
              (top *. (1.0 -. wy)) +. (bot *. wy))))
