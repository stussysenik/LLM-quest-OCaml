"""Export an image and its preprocessing steps for the OCaml `Image` gates.

Three steps, gated at three different levels of honesty, and the export
prints which is which:

  1. `to_tensor`            uint8 [H][W][C] -> float [C][H][W] / 255.
                            Pure arithmetic. Reproduced exactly. 1e-9.
  2. `normalize`            per-channel (x - mean) / std, ImageNet.
                            Pure arithmetic. Reproduced exactly. 1e-9.
  3. `resize_bilinear`      torch's `interpolate(mode="bilinear",
                            align_corners=False)` sample positions.

**Step 3 is not torchvision's resize and this script does not pretend it
is.** `vlm_generation.py` calls
`torchvision.transforms.functional.resize`, which antialiases and,
because it operates on a PIL image, resamples in the *integer* domain and
rounds. Antialiasing is a filter-bank specification, not a formula, and
reproducing it to 1e-9 is a project rather than a gate. What is gated
here is the sample-position rule, against torch's own resampler.

That is worth doing anyway: with steps 1 and 2 exact and step 3 known to
be the single non-exact stage, a downstream divergence has one candidate
cause instead of three. The alternative — skip preprocessing and let the
caller hand the model a tensor — leaves the layout and the normalisation
constants ungated, which is where the quieter bugs live.

Run:
    <env>/bin/python ocaml/parity/export_image.py <out>.safetensors
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pinned

pinned.check_pinned()

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

SEED = 5500
IMG_H, IMG_W = 9, 13          # deliberately not square, and not a
OUT = 16                        # nice ratio, to catch axis mix-ups
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main(out_path):
    g = torch.Generator().manual_seed(SEED)
    print("module: image preprocessing (to_tensor, normalize, resize_bilinear)")
    print(f"exact steps: to_tensor (/255), ImageNet normalize — gated 1e-9")
    print(f"approximate step: resize = torch interpolate(bilinear, align_corners=False); "
          f"NOT torchvision F.resize (antialias + uint8 rounding) — gated at 1e-6")
    print(f"shape: {IMG_H}x{IMG_W} -> {OUT}x{OUT}, channels 3 (RGB)")
    print(pinned.banner())

    # Raw uint8 pixels, full range on purpose: a 0..1 image would make a
    # /255 mistake invisible.
    px = torch.randint(0, 256, (IMG_H, IMG_W, 3), generator=g, dtype=torch.int64)

    t = {}
    # Stored flat: the [H][W][C] -> nested-array reshape is the OCaml
    # side's job and is what gate I-a is partly about, so the reference
    # should not hand over a shape that does it for us.
    t["in.px"] = px.reshape(-1).clone().to(torch.int64)

    # Step 2: to_tensor. float [C][H][W] / 255.
    as_tensor = px.permute(2, 0, 1).double() / 255.0
    t["ref.to_tensor"] = as_tensor.clone().contiguous()

    # Step 3: ImageNet normalize, applied to the *same* tensor.
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float64).reshape(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float64).reshape(-1, 1, 1)
    t["ref.normalized"] = ((as_tensor - mean) / std).clone().contiguous()

    # Step 1: resize. Input as float [1][C][H][W], NCHW as interpolate wants.
    # Resizing the *float* tensor rather than the uint8 one is the
    # documented difference from torchvision; doing it in float also means
    # the gate is about sample positions, not about rounding.
    resized = F.interpolate(
        as_tensor.unsqueeze(0), size=(OUT, OUT), mode="bilinear", align_corners=False
    ).squeeze(0)
    t["ref.resized"] = resized.clone().contiguous()
    # And normalized-after-resized, which is the order the real pipeline
    # uses once the resize is moved into float.
    t["ref.resized_normalized"] = ((resized - mean) / std).clone().contiguous()

    save_file(t, out_path)
    print(f"\nto_tensor range: [{as_tensor.min():.6f}, {as_tensor.max():.6f}] "
          f"(expect [0.0, 1.0])")
    print(f"normalize range: [{t['ref.normalized'].min():.4f}, {t['ref.normalized'].max():.4f}]")
    print(f"resize range: [{resized.min():.6f}, {resized.max():.6f}] "
          f"(a bilinear resample of [0,1] stays in [0,1])")
    print(f"saved {out_path} ({len(t)} tensors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="/tmp/ocaml-parity/image_ref.safetensors")
    main(ap.parse_args().out)
