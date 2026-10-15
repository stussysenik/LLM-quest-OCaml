"""Export ViT reference values for the OCaml multimodal-Part-1 gate.

Same contract as the other exports: the reference comes from instantiating
the repo's own modules with deterministic weights, and the inputs are
written alongside the outputs so the OCaml gate cannot be comparing its own
idea of the inputs against PyTorch's.

The ViT is verified at a small shape (emb 32, 3 layers, 2 heads, 8x8 image
with 2x2 patches = 16 patches) rather than at VIT_BASE_CONFIG, for the same
reason D2 uses tiny and mid cases: a big model makes a structural bug look
like a numerical one, and it is slow. Two shapes are exported so that the
16x16-token and 17x17-token cases (with and without the CLS token) are both
covered and `pos_embedding` is length-checked against the sequence it is
added to.

Weights are randn / sqrt(fan_in) so activations stay O(1); dropout is
identity in eval mode regardless of the configured rate, which is what
`model.eval()` gives HF and what the OCaml port assumes.

Writes (all F64 — see safetensors.ml mat_of_f64: the gate's tolerance is
bounded by the less precise side, and fp32 is not that side here):
  v1.<case>.img         [C, H, W]
  v1.<case>.conv_weight [emb, C * P * P]
  v1.<case>.conv_bias   [emb]
  v1.<case>.cls_token   [emb]
  v1.<case>.pos_embed   [num_patches + 1, emb]
  v1.<case>.block{i}.ln1_scale / ln1_shift   [emb]
  v1.<case>.block{i}.ln2_scale / ln2_shift   [emb]
  v1.<case>.block{i}.att_{wq,wk,wv,wo}.{weight,bias}
  v1.<case>.block{i}.ffn_{l1,l2}.{weight,bias}
  v1.<case>.final_scale / final_shift         [emb]
  v1.<case>.classifier.{weight,bias}
  v1.<case>.ref_patch_emb  [num_patches + 1, emb]  — PatchEmbedding2D(img)
  v1.<case>.ref_pos_added  [num_patches + 1, emb]  — patch_embedding + pos_embedding
  v1.<case>.ref_hidden  [num_patches + 1, emb]  — ViTModel(..., output_hidden_states=True)
  v1.<case>.ref_logits  [num_classes]           — ViTModel(...)

Usage:
    python ocaml/parity/export_vit.py [out.safetensors]

Pinned stack: see parity/requirements.txt + parity/pinned.py.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_quest.multimodal.vision_transformer.vit_model import ViTModel

import pinned

pinned.check_pinned()

SEED = 20260101
# (name, img_h, img_w, patch, channels, emb, layers, heads, classes)
CASES = {
    "small": dict(img_height=8, img_width=8, patch_size=2, num_channels=3, emb_dim=32, n_layers=3, n_heads=2, num_classes=5, drop_rate=0.0, qkv_bias=True),
    # Square, because ViTModel.forward asserts x.shape[2] == img_width while
    # the tensor is (b, C, H, W) — so dim 2 is height. The assert only agrees
    # with the conv below it when H == W. Recorded rather than worked around:
    # the port must mirror the reference, and the reference only runs on
    # square images. This case still differs structurally from "small"
    # (patch 3 not 2, so 4 patches not 16; 4 heads not 2).
    "odd": dict(img_height=6, img_width=6, patch_size=3, num_channels=3, emb_dim=48, n_layers=2, n_heads=4, num_classes=7, drop_rate=0.0, qkv_bias=True),
}


def randn_like(out_rows, in_cols, g):
    return torch.randn(out_rows, in_cols, generator=g) / math.sqrt(in_cols)


def export_case(name, cfg, tensors):
    g = torch.Generator().manual_seed(SEED + cfg["emb_dim"])
    emb, C, P = cfg["emb_dim"], cfg["num_channels"], cfg["patch_size"]
    H, W = cfg["img_height"], cfg["img_width"]
    n_patch = (H * W) // (P * P)

    vit = ViTModel(cfg)
    with torch.no_grad():
        pe = vit.patch_embedding
        vit.patch_embedding.conv_proj.weight.copy_(
            randn_like(emb, C * P * P, g).reshape(emb, C, P, P)
        )
        vit.patch_embedding.conv_proj.bias.copy_(randn_like(emb, 1, g).reshape(-1))
        vit.patch_embedding.cls_token.copy_(randn_like(1, emb, g).reshape(1, 1, emb))
        vit.pos_embedding.copy_(randn_like(n_patch + 1, emb, g).reshape(1, n_patch + 1, emb))
        for blk in vit.transformer_blocks:
            blk.ln_1.scale.copy_(randn_like(emb, 1, g).reshape(-1))
            blk.ln_1.shift.copy_(randn_like(emb, 1, g).reshape(-1))
            blk.ln_2.scale.copy_(randn_like(emb, 1, g).reshape(-1))
            blk.ln_2.shift.copy_(randn_like(emb, 1, g).reshape(-1))
            for name_, lin in (
                ("w_queries", blk.att.w_queries),
                ("w_keys", blk.att.w_keys),
                ("w_values", blk.att.w_values),
                ("out_proj", blk.att.out_proj),
            ):
                lin.weight.copy_(randn_like(lin.weight.shape[0], lin.weight.shape[1], g))
                lin.bias.copy_(randn_like(lin.bias.shape[0], 1, g).reshape(-1))
            for lin in (blk.ffn.layers[0], blk.ffn.layers[2]):
                lin.weight.copy_(randn_like(lin.weight.shape[0], lin.weight.shape[1], g))
                lin.bias.copy_(randn_like(lin.bias.shape[0], 1, g).reshape(-1))
        vit.final_ln.scale.copy_(randn_like(emb, 1, g).reshape(-1))
        vit.final_ln.shift.copy_(randn_like(emb, 1, g).reshape(-1))
        vit.classifier.weight.copy_(randn_like(cfg["num_classes"], emb, g))
        vit.classifier.bias.copy_(randn_like(cfg["num_classes"], 1, g).reshape(-1))

    vit = vit.double().eval()
    img = torch.randn(1, C, H, W, generator=g).double()

    with torch.no_grad():
        ref_patch_emb = vit.patch_embedding(img)[0].cpu().clone().contiguous()
        ref_pos = (vit.patch_embedding(img) + vit.pos_embedding)[0].cpu().clone().contiguous()
        ref_hidden = vit(img, output_hidden_states=True)[0].cpu().clone().contiguous()
        ref_logits = vit(img)[0].cpu().clone().contiguous()

    t = tensors
    t[f"v1.{name}.img"] = img[0].cpu().clone().contiguous()
    t[f"v1.{name}.conv_weight"] = (
        vit.patch_embedding.conv_proj.weight.reshape(emb, C * P * P).cpu().clone().contiguous()
    )
    t[f"v1.{name}.conv_bias"] = vit.patch_embedding.conv_proj.bias.cpu().clone().contiguous()
    t[f"v1.{name}.cls_token"] = vit.patch_embedding.cls_token.reshape(-1).cpu().clone().contiguous()
    t[f"v1.{name}.pos_embed"] = vit.pos_embedding.reshape(n_patch + 1, emb).cpu().clone().contiguous()
    for i, blk in enumerate(vit.transformer_blocks):
        p = f"v1.{name}.block{i}"
        t[f"{p}.ln1_scale"] = blk.ln_1.scale.cpu().clone().contiguous()
        t[f"{p}.ln1_shift"] = blk.ln_1.shift.cpu().clone().contiguous()
        t[f"{p}.ln2_scale"] = blk.ln_2.scale.cpu().clone().contiguous()
        t[f"{p}.ln2_shift"] = blk.ln_2.shift.cpu().clone().contiguous()
        for key, lin in (
            ("wq", blk.att.w_queries),
            ("wk", blk.att.w_keys),
            ("wv", blk.att.w_values),
            ("wo", blk.att.out_proj),
        ):
            t[f"{p}.att_{key}_weight"] = lin.weight.cpu().clone().contiguous()
            t[f"{p}.att_{key}_bias"] = lin.bias.cpu().clone().contiguous()
        for key, lin in (("l1", blk.ffn.layers[0]), ("l2", blk.ffn.layers[2])):
            t[f"{p}.ffn_{key}_weight"] = lin.weight.cpu().clone().contiguous()
            t[f"{p}.ffn_{key}_bias"] = lin.bias.cpu().clone().contiguous()
    t[f"v1.{name}.final_scale"] = vit.final_ln.scale.cpu().clone().contiguous()
    t[f"v1.{name}.final_shift"] = vit.final_ln.shift.cpu().clone().contiguous()
    t[f"v1.{name}.classifier_weight"] = vit.classifier.weight.cpu().clone().contiguous()
    t[f"v1.{name}.classifier_bias"] = vit.classifier.bias.cpu().clone().contiguous()
    t[f"v1.{name}.ref_patch_emb"] = ref_patch_emb
    t[f"v1.{name}.ref_pos_added"] = ref_pos
    t[f"v1.{name}.ref_hidden"] = ref_hidden
    t[f"v1.{name}.ref_logits"] = ref_logits

    print(
        f"{name}: img{tuple(img.shape)} -> hidden{tuple(ref_hidden.shape)} "
        f"logits{tuple(ref_logits.shape)}  patches={n_patch}  |hidden|max={ref_hidden.abs().max():.4f}"
    )


def main(out_path):
    print(f"module: {ViTModel.__module__}.{ViTModel.__name__}")
    print(
        "conventions: conv2d stride=kernel (patch projection), [emb, C*P*P] kernel; "
        "row-major patch order over the (H/P, W/P) grid; CLS prepended at position 0; "
        "unmasked (encoder) attention; LayerNorm is (x-mean)/(std+eps) — eps OUTSIDE the sqrt"
    )
    print(f"inputs: seed={SEED}")
    print(pinned.banner())

    tensors = {}
    for name, cfg in CASES.items():
        export_case(name, cfg, tensors)

    # A GELU grid, because the ViT FFN is the only place in this port where
    # GELU is compared against torch's *exact* erf. The OCaml kernel computes
    # erf itself (Maclaurin series below 2, continued fraction above; 1.11e-15
    # measured over [-12, 115]), so the grid pins the V-g gate at 1e-13 — far
    # above the port's own ULP of drift and far below the 2.12e-07 bound of
    # the Abramowitz & Stegun polynomial it replaced.
    gx = torch.tensor([-8.0, -4.0, -3.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0], dtype=torch.float64)
    tensors["v1.gelu.x"] = gx
    tensors["v1.gelu.ref"] = torch.nn.functional.gelu(gx).contiguous()

    save_file(tensors, out_path)
    print(f"saved {out_path} ({len(tensors)} tensors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="/tmp/ocaml-parity/vit_ref.safetensors")
    main(ap.parse_args().out)