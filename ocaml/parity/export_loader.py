"""Export a fused VLM checkpoint in the exact namespace layout `Vlm_loader` reads.

Why this script exists separately from `export_vlm.py`
-----------------------------------------------------
`export_vlm.py` gates the *forward pass*: it flattens the three models into
convenient names (`vit.conv_weight`, `gpt.tok_emb`, …) and stores
reference activations beside them. Those names are chosen for readability
inside a test, and they are **not** the names in any real checkpoint.

This script writes what `vlm_training.py` would actually save: each
model's own `state_dict()`, verbatim, under three namespaces in one file —

    vit.<ViTModel state_dict key>
    adapter.<ViTAdapter state_dict key>
    gpt.<GPTModel state_dict key>

— which is precisely what `Vlm_loader`/`Vit_loader`/`Gpt_repo_loader` claim
via `Safetensors.subset`. Two things follow that are the whole point:

  * the loader's `expected_keys` is checked against a *real* `state_dict`,
    not against a hand-written table of what someone believed the keys
    were. A key the loader invents fails here, loudly, before the first
    parity number is read.
  * the reference activations are produced by the same live modules whose
    weights were written, so a gate cannot pass by comparing the port
    against its own idea of the inputs.

Deliberately *not* here: the dropout in the adapter's ffn branch. It is
`nn.Dropout`, holds no tensor at `p=0.0`, and therefore is not in the
`state_dict` — so the loader must not claim it. Getting that wrong is the
kind of mistake a real checkpoint finds and a hand-written table does not.

Run:
    <env>/bin/python ocaml/parity/export_loader.py <out>.safetensors
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pinned

pinned.check_pinned()

import torch
from safetensors.torch import save_file

from llm_quest.gpt.gpt_model import GPTModel
from llm_quest.multimodal.vision_transformer.vit_model import ViTModel
from llm_quest.multimodal.vision_transformer.vit_engine import ViTAdapter

# Must match `Config.gpt_tiny_test_config` *except* n_positions: the tiny
# config declares 16, and a ViT image alone is 17 tokens (64 patches +
# CLS at 8x8/patch2), so a 16-position decoder cannot hold even the image
# with no text at all. 64 leaves room for the fused sequence. Getting this
# wrong shows up as a mask/shape error rather than a silent wrong answer,
# which is why it is stated here rather than discovered there.
SEED = 4100
VIT = dict(
    img_width=8, img_height=8, patch_size=2, num_channels=3, emb_dim=32,
    n_layers=3, n_heads=2, num_classes=5, drop_rate=0.0, qkv_bias=True,
)
GPT = dict(
    emb_dim=16, vocab_size=64, context_length=64, n_layers=2, n_heads=2,
    drop_rate=0.0, qkv_bias=False, ffn_hidden_dim=None,
)
TEXT_LEN = 6
ADAPTER_TYPE = "ffn"
HIDDEN_SIZE_FACTOR = 4

# Generation reference. `MAX_GEN` is 12 rather than the reference's 70:
# a tiny random model rarely emits EOS, so the run would go the distance
# and export 70 rows of logits to check a few. 12 steps is enough to
# cross several decode steps and the cache boundary.
MAX_GEN = 12
MAX_GATED_STEPS = 7
# The loop runs a fixed 12 steps: a randomly-initialised model's greedy
# argmax almost never lands on the VLM's EOS, so an EOS branch here would
# be dead code that reads like a promise. The stop contract is gated by
# the generation tests, not by this export.


def build_vit(g):
    torch.manual_seed(SEED)
    vit = ViTModel(VIT)
    with torch.no_grad():
        for p in vit.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64))
    return vit.double().eval()


def build_gpt(g):
    torch.manual_seed(SEED + 1)
    gpt = GPTModel(GPT)
    with torch.no_grad():
        for p in gpt.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64))
    # `trf_blocks.N.att.mask` is a *buffer*, not a parameter, so the loop
    # above never touched it. It is still in the state_dict and the loader
    # still claims it, so leave the real causal mask in place rather than
    # fabricating one.
    return gpt.double().eval()


def build_adapter(g):
    torch.manual_seed(SEED + 2)
    adapter = ViTAdapter(
        vit_d_out=VIT["emb_dim"], llm_d_in=GPT["emb_dim"],
        adapter_type=ADAPTER_TYPE, hidden_size_factor=HIDDEN_SIZE_FACTOR,
    )
    with torch.no_grad():
        for p in adapter.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64))
    # ViTAdapter's default is bias=False and both VLM scripts use it; the
    # loader therefore claims no `*.bias` for the adapter, and this asserts
    # the file agrees before the gate can be blamed for a mismatch.
    for name, p in adapter.named_parameters():
        assert p.dim() == 2, f"unexpected adapter parameter {name}"
    return adapter.double().eval()


def main(out_path):
    g = torch.Generator().manual_seed(SEED)
    print("module: ViTModel + ViTAdapter + GPTModel, raw state_dict names")
    print(f"namespaces: vit. ({VIT['n_layers']} blocks)  adapter. ({ADAPTER_TYPE})  gpt. ({GPT['n_layers']} blocks)")
    print(f"conventions: ViT LayerNorm eps OUTSIDE sqrt; ViTAdapter bias-free, exact GELU; "
          f"GPT LayerNorm eps OUTSIDE sqrt, exact GELU")
    print(f"widths: vit_emb={VIT['emb_dim']} -> llm_emb={GPT['emb_dim']} (deliberately unequal)")
    print(pinned.banner())

    vit = build_vit(g)
    gpt = build_gpt(g)
    adapter = build_adapter(g)

    t = {}
    # --- the checkpoint itself, verbatim state_dict names -------------------
    # Weights go out as F32, references as F64, which is the convention
    # `export_reference.py` sets and the reason for it: a *checkpoint* is
    # float32 on disk because that is what `torch.save` writes and what
    # every loader in this project asserts, while the *reference
    # activations* are float64 because the gate compares two
    # implementations of one function and the tolerance it can hold is set
    # by the less precise of the two. Writing weights as F64 here would
    # make the loaders' F32 assertion a special case that only this
    # checkpoint triggers — which is how a real F32 checkpoint would end
    # up un-loadable and nobody would notice until someone had one.
    for k, v in vit.state_dict().items():
        t[f"vit.{k}"] = v.to(torch.float32).clone().contiguous()
    for k, v in adapter.state_dict().items():
        t[f"adapter.{k}"] = v.to(torch.float32).clone().contiguous()
    for k, v in gpt.state_dict().items():
        # `trf_blocks.N.att.mask` is a *bool* buffer, and it is the one
        # tensor deliberately left at its native dtype: `safetensors.torch`
        # writes BOOL as one byte per element, so a real converted
        # checkpoint carries it as BOOL. Casting it to F32 would hide the
        # fact that the reader must understand BOOL at all — and a reader
        # that does not is a reader that cannot open the file.
        if k.endswith("att.mask"):
            t[f"gpt.{k}"] = v.clone().contiguous()
        else:
            t[f"gpt.{k}"] = v.to(torch.float32).clone().contiguous()

    C, H, W = VIT["num_channels"], VIT["img_height"], VIT["img_width"]
    P = VIT["patch_size"]
    n_patch = (H * W) // (P * P)
    n_vit = n_patch + 1

    img = torch.randn(1, C, H, W, generator=g, dtype=torch.float64)
    ids = torch.randint(0, GPT["vocab_size"], (1, TEXT_LEN), generator=g)

    with torch.no_grad():
        vit_hidden = vit(img, output_hidden_states=True)
        vision = adapter(vit_hidden)
        text_emb = gpt.emb_dict(ids) + gpt.pos_emb_dict(torch.arange(TEXT_LEN).unsqueeze(0))
        fused = torch.cat([vision, text_emb], dim=1)
        logits = gpt(fused, input_embedded=True)
        # The ViT alone, for Vit_loader's own gate.
        vit_logits = vit(img)

    t["in.img"] = img[0].clone().contiguous()
    t["in.input_ids"] = ids[0].clone().contiguous().to(torch.int64)
    t["ref.vit_hidden"] = vit_hidden[0].clone()
    t["ref.vision"] = vision[0].clone()
    t["ref.fused"] = fused[0].clone()
    t["ref.logits"] = logits[0].clone()
    t["ref.vit_logits"] = vit_logits.clone()

    # --- generation -------------------------------------------------------
    # `vlm_generation.py`'s loop, reproduced literally:
    #   seq = vision_embeddings            (no prompt, no positions)
    #   loop: logits = gpt(seq, input_embedded=True)[:, -1, :]
    #         tok = argmax(logits); seq = cat([seq, emb_dict(tok)])
    # Note what is NOT here: `get_embeddings`, and therefore the extra
    # positional the *training* path bakes into the text embeddings. The
    # generation loop appends `emb_dict(tok)` bare and lets
    # `input_embedded=True` place every row exactly once. A port that
    # builds this sequence with `Vlm.fused_embeddings` double-counts
    # positions and still produces plausible captions, which is why the
    # gate below anchors on these tokens specifically.
    with torch.no_grad():
        seq = vision
        greedy = []
        step_logits = []
        for _ in range(MAX_GEN):
            lg = gpt(seq, input_embedded=True)[0, -1, :]
            step_logits.append(lg.clone())
            tok = int(torch.argmax(lg).item())
            greedy.append(tok)
            seq = torch.cat([seq, gpt.emb_dict.weight[tok].reshape(1, 1, -1)], dim=1)

    t["ref.vision_seq"] = seq[0, :n_vit, :].clone()      # the prefill window
    t["ref.prefill_last_logits"] = step_logits[0].clone()
    t["ref.greedy_tokens"] = torch.tensor(greedy, dtype=torch.int64)
    # Logits for the first few decode steps, so a gate can compare the
    # *cached* path against the *uncached* one row by row rather than
    # only through the argmax, which would hide a small error.
    for i, lg in enumerate(step_logits[:MAX_GATED_STEPS]):
        t[f"ref.step{i}_last_logits"] = lg.clone()

    print(f"\ngeneration: {len(greedy)} greedy tokens (fixed {MAX_GEN}-step run; "
          f"EOS stop is gated by test_loader_vlm G-d)")

    n_conv = len([k for k in vit.state_dict() if "conv_proj" in k])
    print(f"\nviT state_dict: {len(vit.state_dict())} tensors (conv tensors are 4-D: {n_conv})")
    print(f"adapter state_dict: {len(adapter.state_dict())} tensors, keys {sorted(adapter.state_dict())}")
    print(f"gpt state_dict: {len(gpt.state_dict())} tensors "
          f"(incl. {len([k for k in gpt.state_dict() if k.endswith('att.mask')])} mask buffers)")
    print(f"fused sequence: {n_vit} vision + {TEXT_LEN} text = {fused.shape[1]} "
          f"(decoder context {GPT['context_length']})")

    save_file(t, out_path)
    print(f"saved {out_path} ({len(t)} tensors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="/tmp/ocaml-parity/vlm_loader_ref.safetensors")
    main(ap.parse_args().out)
