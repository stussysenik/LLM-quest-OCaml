"""Export VLM reference values for the OCaml multimodal-Part-2 gate.

Part 2 is [ViTAdapter] + early fusion ([vlm_engine.py]). The reference is
the repo's own `ViTModel` + `ViTAdapter` + `GPTModel` wired exactly as
`vlm_training._calc_loss_batch_vlm` wires them, with deterministic
weights.

Two adapter types are exported, because both are real:
  - "ffn"    — what `vlm_generation.py` and `vlm_training.py` construct
  - "simple" — the other branch of `ViTAdapter.__init__`

The ViT and the LLM deliberately have **different** widths (32 vs 48).
Exporting a same-width pair would let an identity adapter pass: the
whole point of the adapter is a projection, and a square case cannot
distinguish "projected" from "passed through".

The ViT's classification head is not exported — the multimodal README
drops it, since the VLM only wants `output_hidden_states`, i.e.
`forward_hidden_states`, never the CLS classifier.

Writes (all F64 except the integer inputs — see safetensors.ml
mat_of_f64: the gate's tolerance is bounded by the less precise side):
  v2.<case>.img                      [C, H, W]
  v2.<case>.vit.conv_weight          [vit_emb, C * P * P]
  v2.<case>.vit.conv_bias            [vit_emb]
  v2.<case>.vit.cls_token            [vit_emb]
  v2.<case>.vit.pos_embed            [num_patches + 1, vit_emb]
  v2.<case>.vit.block{i}.ln1_scale / ln1_shift / ln2_scale / ln2_shift
  v2.<case>.vit.block{i}.att_{wq,wk,wv,wo}_{weight,bias}
  v2.<case>.vit.block{i}.ffn_{l1,l2}_{weight,bias}
  v2.<case>.vit.final_scale / final_shift
  v2.<case>.adapter_l1_weight        [l1_out, vit_emb]
  v2.<case>.adapter_l2_weight        [llm_emb, hidden]        (ffn only)
  v2.<case>.gpt.tok_emb              [vocab, llm_emb]
  v2.<case>.gpt.pos_emb              [context_length, llm_emb]
  v2.<case>.gpt.block{i}.*           — as above, GPT-2 shapes
  v2.<case>.gpt.final_scale / final_shift
  v2.<case>.gpt.out_weight           [vocab, llm_emb]
  v2.<case>.input_ids                [T]            I64
  v2.<case>.text_mask                [T]            I64 (1 = real)
  v2.<case>.ref_vit_hidden           [num_patches + 1, vit_emb]
  v2.<case>.ref_vision               [num_patches + 1, llm_emb]  — adapter output
  v2.<case>.ref_text_emb             [T, llm_emb]               — token + position
  v2.<case>.ref_fused                [num_patches + 1 + T, llm_emb]
  v2.<case>.ref_logits               [num_patches + 1 + T, vocab]

Usage:
    python ocaml/parity/export_vlm.py [out.safetensors]

Pinned stack: see parity/requirements.txt + parity/pinned.py.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_quest.gpt.gpt_model import GPTModel
from llm_quest.multimodal.vision_transformer.vit_engine import ViTAdapter
from llm_quest.multimodal.vision_transformer.vit_model import ViTModel

import pinned

pinned.check_pinned()

SEED = 20260102

# Shared backbone shape. vit_emb != llm_emb on purpose (see module doc).
VIT = dict(img_height=8, img_width=8, patch_size=2, num_channels=3, emb_dim=32, n_layers=2, n_heads=2, drop_rate=0.0, qkv_bias=True, num_classes=5)
GPT = dict(vocab_size=101, context_length=64, emb_dim=48, n_layers=2, n_heads=2, drop_rate=0.0, qkv_bias=True)
TEXT_LEN = 6  # right-padded: last 3 are padding

CASES = {"ffn": "ffn", "simple": "simple"}


def randn(out_rows, in_cols, g):
    return torch.randn(out_rows, in_cols, generator=g) / math.sqrt(in_cols)


def randn_vec(n, g):
    return torch.randn(n, generator=g) / math.sqrt(n)


def build_vit(cfg, g):
    """A ViT with every weight drawn randn/sqrt(fan_in), as export_vit does."""
    vit = ViTModel(cfg)
    emb, C, P = cfg["emb_dim"], cfg["num_channels"], cfg["patch_size"]
    n_patch = (cfg["img_height"] * cfg["img_width"]) // (P * P)
    with torch.no_grad():
        vit.patch_embedding.conv_proj.weight.copy_(randn(emb, C * P * P, g).reshape(emb, C, P, P))
        vit.patch_embedding.conv_proj.bias.copy_(torch.zeros(emb))
        vit.patch_embedding.cls_token.copy_(randn(1, emb, g).reshape(1, 1, emb))
        vit.pos_embedding.copy_(randn(n_patch + 1, emb, g).reshape(1, n_patch + 1, emb))
        for blk in vit.transformer_blocks:
            for ln in (blk.ln_1, blk.ln_2):
                ln.scale.copy_(randn_vec(emb, g))
                ln.shift.copy_(randn_vec(emb, g))
            for lin in (blk.att.w_queries, blk.att.w_keys, blk.att.w_values, blk.att.out_proj):
                lin.weight.copy_(randn(lin.weight.shape[0], lin.weight.shape[1], g))
                lin.bias.copy_(randn_vec(lin.bias.shape[0], g))
            for lin in (blk.ffn.layers[0], blk.ffn.layers[2]):
                lin.weight.copy_(randn(lin.weight.shape[0], lin.weight.shape[1], g))
                lin.bias.copy_(randn_vec(lin.bias.shape[0], g))
        vit.final_ln.scale.copy_(randn_vec(emb, g))
        vit.final_ln.shift.copy_(randn_vec(emb, g))
    return vit


def build_gpt(cfg, g):
    """A GPT-2 with every weight drawn randn/sqrt(fan_in).

    GPT-2's own LayerNorm is (x-mean)/(std+eps) — eps OUTSIDE the sqrt,
    same as ViT — and its FFN uses the exact-erf GELU, so the OCaml side
    must build its blocks with Eps_outside_sqrt / Gelu_exact.
    """
    gpt = GPTModel(cfg)
    e = cfg["emb_dim"]
    inner = 4 * e
    with torch.no_grad():
        gpt.emb_dict.weight.copy_(randn(cfg["vocab_size"], e, g))
        gpt.pos_emb_dict.weight.copy_(randn(cfg["context_length"], e, g))
        for blk in gpt.trf_blocks:
            for ln in (blk.ln_1, blk.ln_2):
                ln.scale.copy_(randn_vec(e, g))
                ln.shift.copy_(randn_vec(e, g))
            for lin in (blk.att.w_queries, blk.att.w_keys, blk.att.w_values, blk.att.out_proj):
                lin.weight.copy_(randn(lin.weight.shape[0], lin.weight.shape[1], g))
                if lin.bias is not None:
                    lin.bias.copy_(randn_vec(lin.bias.shape[0], g))
            for lin in (blk.ffn.layers[0], blk.ffn.layers[2]):
                lin.weight.copy_(randn(lin.weight.shape[0], lin.weight.shape[1], g))
                lin.bias.copy_(randn_vec(lin.bias.shape[0], g))
        gpt.final_ln.scale.copy_(randn_vec(e, g))
        gpt.final_ln.shift.copy_(randn_vec(e, g))
        gpt.out.weight.copy_(randn(cfg["vocab_size"], e, g))
    return gpt


def store_vit_weights(t, p, vit):
    # Prefixed `vit.` — GPT-2 has blocks and a final LN too, and an
    # unprefixed `{p}.block0.ln1_scale` would be written by both store
    # functions, with the second silently clobbering the first.
    p = f"{p}.vit"
    emb, C, P = VIT["emb_dim"], VIT["num_channels"], VIT["patch_size"]
    t[f"{p}.conv_weight"] = vit.patch_embedding.conv_proj.weight.reshape(emb, C * P * P).clone()
    t[f"{p}.conv_bias"] = vit.patch_embedding.conv_proj.bias.clone()
    t[f"{p}.cls_token"] = vit.patch_embedding.cls_token.reshape(-1).clone()
    n_patch = (VIT["img_height"] * VIT["img_width"]) // (P * P)
    t[f"{p}.pos_embed"] = vit.pos_embedding.reshape(n_patch + 1, emb).clone()
    for i, blk in enumerate(vit.transformer_blocks):
        q = f"{p}.block{i}"
        for name, ln in (("ln1", blk.ln_1), ("ln2", blk.ln_2)):
            t[f"{q}.{name}_scale"] = ln.scale.clone()
            t[f"{q}.{name}_shift"] = ln.shift.clone()
        for key, lin in (("wq", blk.att.w_queries), ("wk", blk.att.w_keys), ("wv", blk.att.w_values), ("wo", blk.att.out_proj)):
            t[f"{q}.att_{key}_weight"] = lin.weight.clone()
            t[f"{q}.att_{key}_bias"] = lin.bias.clone()
        for key, lin in (("l1", blk.ffn.layers[0]), ("l2", blk.ffn.layers[2])):
            t[f"{q}.ffn_{key}_weight"] = lin.weight.clone()
            t[f"{q}.ffn_{key}_bias"] = lin.bias.clone()
    t[f"{p}.final_scale"] = vit.final_ln.scale.clone()
    t[f"{p}.final_shift"] = vit.final_ln.shift.clone()


def store_gpt_weights(t, p, gpt):
    # Prefixed `gpt.` for the same reason `store_vit_weights` prefixes
    # `vit.`: both models have `block{i}` and `final_{scale,shift}`.
    p = f"{p}.gpt"
    e = GPT["emb_dim"]
    t[f"{p}.tok_emb"] = gpt.emb_dict.weight.clone()
    t[f"{p}.pos_emb"] = gpt.pos_emb_dict.weight.clone()
    for i, blk in enumerate(gpt.trf_blocks):
        q = f"{p}.block{i}"
        for name, ln in (("ln1", blk.ln_1), ("ln2", blk.ln_2)):
            t[f"{q}.{name}_scale"] = ln.scale.clone()
            t[f"{q}.{name}_shift"] = ln.shift.clone()
        for key, lin in (("wq", blk.att.w_queries), ("wk", blk.att.w_keys), ("wv", blk.att.w_values), ("wo", blk.att.out_proj)):
            t[f"{q}.att_{key}_weight"] = lin.weight.clone()
            if lin.bias is not None:
                t[f"{q}.att_{key}_bias"] = lin.bias.clone()
        for key, lin in (("l1", blk.ffn.layers[0]), ("l2", blk.ffn.layers[2])):
            t[f"{q}.ffn_{key}_weight"] = lin.weight.clone()
            t[f"{q}.ffn_{key}_bias"] = lin.bias.clone()
    t[f"{p}.final_scale"] = gpt.final_ln.scale.clone()
    t[f"{p}.final_shift"] = gpt.final_ln.shift.clone()
    t[f"{p}.out_weight"] = gpt.out.weight.clone()


def adapter_linears(adapter):
    """The nn.Linear layers of an adapter, in order.

    `list(adapter.adapter)` on an "ffn" adapter yields
    [Linear, GELU, Linear] — nn.Sequential iterates every child, so a
    naive loop tries to read `.weight` off the activation itself.
    """
    if isinstance(adapter.adapter, torch.nn.Linear):
        return [adapter.adapter]
    return [m for m in adapter.adapter if isinstance(m, torch.nn.Linear)]


def export_case(name, adapter_type, seed_offset, tensors):
    g = torch.Generator().manual_seed(SEED + seed_offset)
    vit_cfg = dict(VIT)
    gpt_cfg = dict(GPT)
    P = vit_cfg["patch_size"]
    n_patch = (vit_cfg["img_height"] * vit_cfg["img_width"]) // (P * P)
    n_vit = n_patch + 1

    vit = build_vit(vit_cfg, g).double().eval()
    gpt = build_gpt(gpt_cfg, g).double().eval()
    adapter = ViTAdapter(
        vit_d_out=vit_cfg["emb_dim"], llm_d_in=gpt_cfg["emb_dim"], adapter_type=adapter_type
    ).double().eval()
    with torch.no_grad():
        linears = adapter_linears(adapter)
        for lin in linears:
            lin.weight.copy_(randn(lin.weight.shape[0], lin.weight.shape[1], g))
            # bias=False is ViTAdapter's default and what the repo's VLM
            # scripts construct; asserting it here keeps the export and the
            # port honest about the one family of bias-free linears.
            assert lin.bias is None, "ViTAdapter default is bias=False"
    adapter = adapter.eval()

    C, H, W = vit_cfg["num_channels"], vit_cfg["img_height"], vit_cfg["img_width"]
    img = torch.randn(1, C, H, W, generator=g).double()
    ids = torch.randint(0, gpt_cfg["vocab_size"], (1, TEXT_LEN), generator=g)
    # Right padding: the last 3 of 6 text positions are padding, so the
    # combined mask is [17 real vision] + [3 real, 3 pad] text.
    text_mask = torch.ones(1, TEXT_LEN, dtype=torch.long)
    text_mask[0, TEXT_LEN - 3 :] = 0

    with torch.no_grad():
        # ViTModel.forward returns the hidden tensor directly when
        # output_hidden_states=True (no wrapper, no batch squeeze) — the
        # shape is (1, n_patches + 1, vit_emb).
        vit_hidden = vit(img, output_hidden_states=True)
        vision = adapter(vit_hidden)  # (1, n_vit, llm_emb)
        text_emb = gpt.emb_dict(ids) + gpt.pos_emb_dict(torch.arange(TEXT_LEN).unsqueeze(0))
        fused = torch.cat([vision, text_emb], dim=1)
        combined_mask = torch.cat([torch.ones(1, n_vit, dtype=torch.bool), text_mask.bool()], dim=1)
        logits = gpt(fused, attn_mask=combined_mask, input_embedded=True)

    t = tensors
    p = f"v2.{name}"
    t[f"{p}.img"] = img[0].clone()
    store_vit_weights(t, p, vit)
    linears = adapter_linears(adapter)
    t[f"{p}.adapter_l1_weight"] = linears[0].weight.clone()
    if len(linears) == 2:
        t[f"{p}.adapter_l2_weight"] = linears[1].weight.clone()
    store_gpt_weights(t, p, gpt)
    t[f"{p}.input_ids"] = ids[0].clone().contiguous()
    t[f"{p}.text_mask"] = text_mask[0].clone().contiguous()
    t[f"{p}.ref_vit_hidden"] = vit_hidden[0].clone()
    t[f"{p}.ref_vision"] = vision[0].clone()
    t[f"{p}.ref_text_emb"] = text_emb[0].clone()
    t[f"{p}.ref_fused"] = fused[0].clone()
    t[f"{p}.ref_logits"] = logits[0].clone()

    print(
        f"{name} ({adapter_type}): img{tuple(img.shape)} -> vit_hidden{tuple(vit_hidden.shape)} "
        f"vision{tuple(vision.shape)} fused{tuple(fused.shape)} logits{tuple(logits.shape)} "
        f"|fused|max={fused.abs().max():.4f}"
    )


def main(out_path):
    print("module: ViTModel + ViTAdapter + GPTModel (llm_quest.multimodal)")
    print(
        "conventions: early fusion concatenates vision FIRST then text "
        "(cat([vision, text], dim=1)); ViTAdapter linears are bias=False; "
        "the adapter FFN uses nn.GELU (exact erf); GPT-2 LayerNorm is "
        "(x-mean)/(std+eps) — eps OUTSIDE the sqrt"
    )
    print(f"inputs: seed={SEED}, vit_emb={VIT['emb_dim']} -> llm_emb={GPT['emb_dim']}, text_len={TEXT_LEN} (last 3 padded)")
    print(pinned.banner())

    tensors = {}
    for i, (name, adapter_type) in enumerate(CASES.items()):
        # Seed offset by case index, not by hash(adapter_type): Python salts
        # str hashes per process (PYTHONHASHSEED), which would make the
        # exported weights differ between runs and turn the reference into
        # a moving target.
        export_case(name, adapter_type, i * 17, tensors)

    save_file(tensors, out_path)
    print(f"saved {out_path} ({len(tensors)} tensors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="/tmp/ocaml-parity/vlm_ref.safetensors")
    main(ap.parse_args().out)
