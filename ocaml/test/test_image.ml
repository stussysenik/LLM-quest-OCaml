(* Gates for [Image] — the three preprocessing steps the multimodal path
   needs before a picture can reach the ViT.

   The tolerances are not uniform, and the non-uniformity is the point:

   - I-a [to_tensor] and I-b [normalize] are pure arithmetic. Gated at
     1e-9 against PyTorch, the same gate as everything else in the port,
     because there is no reason for a divide-by-255 or a mean to be
     inexact.
   - I-c [resize_bilinear] runs at 1e-9 and measures 2.2e-16, but the
     number to keep in mind is *what it is compared against*: torch's
     `interpolate(mode="bilinear", align_corners=False)`, not the
     `torchvision.transforms.functional.resize` that
     `vlm_generation.py` calls. That one antialiases and resamples in the
     integer domain. So the gate is exact about the sample-position rule
     and silent about torchvision's filter — a divergence from real
     captions is expected here and is not evidence of a port bug.

   I-d is the anti-gate, and it is the one worth reading. A bilinear
   resampler is symmetric under swapping its two axes when the input is
   square, and this input is 9x13 — so a row/column mix-up produces a
   16x16 image either way and every shape assertion passes. Only a
   comparison against the reference can see it. This is the same square-
   shape hazard as the [o_proj] layout gate, in a different costume.

   I-e pins the constants. An ImageNet mean written in BGR order, or a
   std of [0.225, 0.224, 0.229], is a plausible-looking mistake that
   produces a model which still classifies, and the only way to notice is
   to assert the numbers. *)

open Tensor

let passed = ref 0
let failed = ref 0

let check name ok =
  if ok then (
    incr passed;
    Printf.printf "PASS %s\n" name)
  else (
    incr failed;
    Printf.printf "FAIL %s\n" name)

let max_abs (a : Vit.image) (b : Vit.image) : float =
  let worst = ref 0.0 in
  Array.iteri
    (fun ci pa ->
      Array.iteri (fun y ra -> Array.iteri (fun x v -> worst := max !worst (abs_float (v -. b.(ci).(y).(x)))) ra) pa)
    a;
  !worst

let gate name (ours : Vit.image) (refm : Vit.image) tol =
  let d = max_abs ours refm in
  Printf.printf "  %-40s max_abs_diff=%.3e\n" name d;
  check name (d <= tol)

let transpose_hw (x : Vit.image) : Vit.image =
  let c = Array.length x in
  let h = Array.length x.(0) in
  let w = Array.length x.(0).(0) in
  (* Deliberately wrong: swap rows and columns. Shape becomes [w][h]. *)
  Array.init c (fun ci -> Array.init w (fun i -> Array.init h (fun j -> x.(ci).(j).(i))))

let () =
  if Array.length Sys.argv <> 2 then (
    Printf.eprintf "usage: %s <image_ref.safetensors>\n" Sys.argv.(0);
    exit 2);
  let f = Safetensors.open_file Sys.argv.(1) in
  Printf.printf "Image preprocessing gates\n\n";

  (* Raw pixels, [H][W][C] ints, exactly as the reference stored them. *)
  let flat = Safetensors.i64_array_of f "in.px" |> Array.map Int64.to_int in
  let h = 9 and w = 13 and c = 3 in
  if Array.length flat <> h * w * c then
    failwith
      (Printf.sprintf "test_image: in.px has %d elements, expected %d" (Array.length flat) (h * w * c));
  let px : Image.raw =
    Array.init h (fun y ->
        Array.init w (fun x -> Array.init c (fun ci -> flat.((((y * w) + x) * c) + ci))))
  in

  Printf.printf "I-a to_tensor: uint8 [H][W][C] -> float [C][H][W] / 255\n";
  let as_tensor = Image.to_tensor px in
  check "I-a layout is [C][H][W]"
    (Array.length as_tensor = c && Array.length as_tensor.(0) = h && Array.length as_tensor.(0).(0) = w);
  gate "I-a vs torch" as_tensor (Safetensors.mat3_of_f64 f "ref.to_tensor") 1e-9;
  (* The divide is 255, not 256. At a saturated pixel the two differ by
     1/65280 = 1.53e-5, four orders above this gate. *)
  let all_white : Image.raw = Array.init h (fun _ -> Array.init w (fun _ -> Array.make c 255)) in
  let white = Image.to_tensor all_white in
  check "I-a 255 maps to exactly 1.0, not 255/256"
    (Array.for_all (fun plane -> Array.for_all (fun row -> Array.for_all (fun v -> v = 1.0) row) plane) white);
  check "I-a 0 maps to exactly 0.0"
    (let black = Image.to_tensor (Array.init h (fun _ -> Array.init w (fun _ -> Array.make c 0))) in
     Array.for_all (fun plane -> Array.for_all (fun row -> Array.for_all (fun v -> v = 0.0) row) plane) black);
  check "I-a an out-of-range pixel is refused"
    (try
       ignore
         (Image.to_tensor
            (Array.init h (fun _ -> Array.init w (fun _ -> [| 0; 0; 300 |]))));
       false
     with Invalid_argument _ -> true);
  check "I-a a non-RGB image is refused"
    (try
       ignore (Image.to_tensor (Array.init h (fun _ -> Array.init w (fun _ -> [| 0; 0 |]))));
       false
     with Invalid_argument _ -> true);

  Printf.printf "\nI-b ImageNet normalize\n";
  gate "I-b vs torch" (Image.normalize_imagenet as_tensor)
    (Safetensors.mat3_of_f64 f "ref.normalized") 1e-9;

  Printf.printf "\nI-e the constants themselves\n";
  check "I-e mean is [0.485; 0.456; 0.406] in RGB order"
    (Image.imagenet_mean = [| 0.485; 0.456; 0.406 |]);
  check "I-e std is [0.229; 0.224; 0.225] in RGB order"
    (Image.imagenet_std = [| 0.229; 0.224; 0.225 |]);
  (* Normalizing the ImageNet mean must give zero — the property that
     pins both the constant and the sign of the operation. *)
  let mean_img =
    Array.init c (fun ci ->
        Array.init h (fun _ -> Array.init w (fun _ -> Image.imagenet_mean.(ci))))
  in
  let mean_norm = Image.normalize_imagenet mean_img in
  check "I-e normalize(mean) == 0"
    (Array.for_all
       (fun plane -> Array.for_all (fun row -> Array.for_all (fun v -> abs_float v < 1e-12) row) plane)
       mean_norm);
  check "I-e normalize is not the identity (it is not just a divide)"
    (let m = Image.normalize_imagenet as_tensor in max_abs m as_tensor > 0.1);

  Printf.printf
    "\nI-c resize_bilinear vs torch interpolate — exact on sample positions,\n     but NOT torchvision's antialiased resize\n";
  let resized = Image.resize_bilinear ~size:16 as_tensor in
  check "I-c output is 16x16"
    (Array.length resized.(0) = 16 && Array.length resized.(0).(0) = 16);
  gate "I-c vs torch interpolate" resized (Safetensors.mat3_of_f64 f "ref.resized") 1e-9;
  gate "I-c normalize-after-resize"
    (Image.normalize_imagenet resized)
    (Safetensors.mat3_of_f64 f "ref.resized_normalized") 1e-9;

  Printf.printf "\nI-d anti-gate: rows/columns swapped must NOT match\n";
  (* The input is 9x13, so a transposed resize has a different value
     layout. Both are 16x16, so nothing but the numbers can tell. *)
  let ref_resized = Safetensors.mat3_of_f64 f "ref.resized" in
  let diff_swapped = max_abs (transpose_hw resized) ref_resized in
  Printf.printf "  %-40s max_abs_diff=%.3e (must exceed 1e-06)\n" "I-d axis swap diverges" diff_swapped;
  check "I-d an axis-swapped resize is rejected" (diff_swapped > 1e-6);
  (* And a half-pixel-off resampler: d * in/out instead of
     (d + 0.5) * in/out - 0.5. Looks right, is a half pixel out. *)
  let naive =
    Array.init c (fun ci ->
        let plane = as_tensor.(ci) in
        Array.init 16 (fun y ->
            let y0 = int_of_float (float_of_int y *. 9.0 /. 16.0) in
            let y0 = if y0 > 8 then 8 else y0 in
            let x0 = int_of_float (float_of_int y *. 13.0 /. 16.0) in
            let x0 = if x0 > 12 then 12 else x0 in
            Array.init 16 (fun x -> plane.(y0).(x0))))
  in
  let diff_naive = max_abs naive ref_resized in
  Printf.printf "  %-40s max_abs_diff=%.3e (must exceed 1e-06)\n" "I-d half-pixel-off diverges"
    diff_naive;
  check "I-d a half-pixel-off resampler is rejected" (diff_naive > 1e-6);
  check "I-d the real resampler does match"
    (max_abs resized ref_resized <= 1e-9);

  Printf.printf "\n%d passed, %d failed\n" !passed !failed;
  exit (if !failed = 0 then 0 else 1)
