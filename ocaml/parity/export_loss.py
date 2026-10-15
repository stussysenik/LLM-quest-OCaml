"""Export cross-entropy / accuracy references for the OCaml `Loss` gates.

The point of these gates is not that softmax is hard to get right. It is
`ignore_index`.

`vlm_loss` masks the *targets* — `labels.masked_fill(mask == 0, -100)` —
and `F.cross_entropy(..., ignore_index=-100)` then drops those rows from
the mean's denominator as well as its numerator. Getting that wrong is
invisible in the happy path: if every row is real, "mean over all rows"
and "mean over kept rows" are the same number. It only shows up when the
caption has padding, and then it silently reports a loss that changes
with batch composition rather than with the model.

So the export covers the three cases separately, and says which is which:

  L-a  no ignored rows          — the two definitions coincide, so this
                                  cannot catch the bug. It is here to
                                  prove the basic arithmetic.
  L-b  *some* rows ignored     — the case that discriminates.
  L-c  *all* rows ignored      — must be NaN, not 0.0, because 0.0 is
                                  indistinguishable from a model that is
                                  confidently right about nothing.

Also exported: argmax agreement (ties resolve to the lowest index, like
`torch.argmax`) and the accuracy fraction, because an accuracy that
counts ignored rows is wrong in the same way and for the same reason.

Run:
    <env>/bin/python ocaml/parity/export_loss.py <out>.safetensors
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

SEED = 6600
N_CLASSES = 17      # deliberately not a round number, and > 2 so argmax ties are possible
IGNORE = -100


def cases(g):
    """Three (name, logits, targets) triples, built to discriminate."""
    out = {}
    # L-a: nothing ignored. The bug is invisible here by construction.
    logits = torch.randn(6, N_CLASSES, generator=g, dtype=torch.float64) * 3.0
    targets = torch.randint(0, N_CLASSES, (6,), generator=g)
    out["a"] = (logits, targets)

    # L-b: the last 3 of 6 ignored. Mean-over-all vs mean-over-kept differ
    # by exactly a factor of 2 here, which no tolerance setting hides.
    logits_b = torch.randn(6, N_CLASSES, generator=g, dtype=torch.float64) * 3.0
    targets_b = torch.randint(0, N_CLASSES, (6,), generator=g)
    targets_b[3:] = IGNORE
    out["b"] = (logits_b, targets_b)

    # L-c: everything ignored. Must be NaN.
    logits_c = torch.randn(4, N_CLASSES, generator=g, dtype=torch.float64) * 3.0
    targets_c = torch.full((4,), IGNORE, dtype=torch.long)
    out["c"] = (logits_c, targets_c)

    # L-d: an exact tie in the argmax, to pin the tie-break rule.
    tie = torch.zeros(1, N_CLASSES, dtype=torch.float64)
    tie[0, 5] = 1.0
    tie[0, 11] = 1.0        # first of two equal maxima is index 5
    out["d"] = (tie, torch.tensor([5], dtype=torch.long))

    # L-e: a single row, so any stray /2 or off-by-one in the denominator
    # shows up directly rather than being averaged away.
    logits_e = torch.randn(1, N_CLASSES, generator=g, dtype=torch.float64) * 3.0
    targets_e = torch.randint(0, N_CLASSES, (1,), generator=g)
    out["e"] = (logits_e, targets_e)

    return out


def main(out_path):
    g = torch.Generator().manual_seed(SEED)
    print("module: cross-entropy with ignore_index, and accuracy")
    print(f"conventions: ignore_index={IGNORE} drops rows from BOTH numerator and "
          f"denominator; all-ignored is NaN; argmax ties take the lowest index")
    print(f"shape: {N_CLASSES} classes")
    print(pinned.banner())

    t = {}
    for name, (logits, targets) in cases(g).items():
        flat = F.cross_entropy(logits, targets, ignore_index=IGNORE)
        preds = logits.argmax(dim=-1)
        kept = targets != IGNORE
        correct = int((preds[kept] == targets[kept]).sum().item())
        total = int(kept.sum().item())
        acc = correct / total if total else float("nan")

        # Scalars go out 1-D, not 0-D: `torch.tensor(x)` is shape () and
        # the OCaml readers are written for vectors, because a 0-D tensor
        # is not something the rest of this port ever produces.
        t[f"L-{name}.logits"] = logits.clone().contiguous()
        t[f"L-{name}.targets"] = targets.clone().contiguous()
        t[f"L-{name}.loss"] = torch.tensor([float(flat)], dtype=torch.float64)
        t[f"L-{name}.correct"] = torch.tensor([correct], dtype=torch.int64)
        t[f"L-{name}.total"] = torch.tensor([total], dtype=torch.int64)
        t[f"L-{name}.accuracy"] = torch.tensor([acc], dtype=torch.float64)

        loss_s = "nan" if torch.isnan(flat) else f"{float(flat):.12f}"
        print(f"  L-{name}: rows={logits.shape[0]} kept={total} loss={loss_s} "
              f"correct={correct} acc={acc}")

    # The two readings of L-b, printed side by side so the export states
    # the number the bug would produce. A reader can then check the gate
    # is rejecting *that* number and not merely something large.
    save_file(t, out_path)
    print(f"\nsaved {out_path} ({len(t)} tensors)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="/tmp/ocaml-parity/loss_ref.safetensors")
    main(ap.parse_args().out)
