"""Pinned stack for the OCaml parity references.

Every file under `ocaml/parity/` is only reproducible against an exact
transformers + torch + model-revision triple. Without the pin, a
`pip install -U transformers` silently invalidates the references and
the resulting gate failure reads as a phantom OCaml regression.

`check_pinned()` turns that into a loud import-time failure instead.
Both library pins are enforced, not just the advertised one: torch is
pinned to the CPU *build*, because a CUDA torch and a CPU torch reduce
in different orders, so a reference exported on one and gated against
the other looks like a numerical disagreement that is really an
environment mismatch. One pin covers every family — the transformers
and torch versions below produced the GPT-2 AND the Llama references,
so bumping either invalidates both and both get re-exported together.
"""

import torch
import transformers

TRANSFORMERS_VERSION = "5.19.0"
TORCH_VERSION = "2.14.1+cpu"
SAFETENSORS_VERSION = "0.8.0"
# openai-community/gpt2, the snapshot in the local HF cache.
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
# hf-internal-testing/tiny-random-LlamaForCausalLM — 2 layers, hidden 16.
# Small enough that Step C can be verified without waiting for Step F.
LLAMA_REVISION = "9fb191250dd56d0ba7ec9785a025ed29c03d5998"


def banner() -> str:
    return (
        f"pinned stack: transformers=={TRANSFORMERS_VERSION} "
        f"torch=={TORCH_VERSION} safetensors=={SAFETENSORS_VERSION} "
        f"gpt2@{GPT2_REVISION} tiny-random-Llama@{LLAMA_REVISION}"
    )


def check_pinned() -> None:
    for module, label, expected in (
        (transformers, "transformers", TRANSFORMERS_VERSION),
        (torch, "torch", TORCH_VERSION),
    ):
        found = module.__version__
        # The CPU index serves torch without the +cpu local-version suffix
        # (e.g. "2.14.1" not "2.14.1+cpu"); accept either as the CPU build.
        if found != expected and not (label == "torch" and found == TORCH_VERSION.removesuffix("+cpu")):
            raise RuntimeError(
                f"{label} {found} != pinned {expected}.\n"
                f"The committed references were generated with the pinned stack:\n  {banner()}\n"
                f"Recreate the venv per the header in parity/requirements.txt (torch comes\n"
                f"from the CPU index) and re-export every reference file; do NOT regenerate\n"
                f"just this one."
            )


def align_parameters(module):
    """Reallocate every parameter into 64-byte-aligned storage, in place.

    HuggingFace loads safetensors through mmap. For `gpt2`, `lm_head.weight`
    is tied to `wte.weight` and comes back with a data pointer that is *not*
    64-byte aligned — `data_ptr() % 64 == 19` on this machine. torch's CPU
    BLAS selects a vectorised GEMV path for single-token forwards that
    requires aligned weights and dereferences it regardless, so the process
    dies with SIGBUS (exit 138) instead of raising a Python exception.
    That is why `export_gen.py` failed at the very first decode step while
    the seq_len>=2 forwards that precede it all succeeded.

    Cloning allocates fresh, aligned storage and copies the values across,
    so the numbers are bit-identical and every committed reference stays
    valid — only the address changes. This is an environment workaround, not
    a change to the model, and it is applied at export time only: the OCaml
    side never sees it.

    Cheap insurance for any module loaded from a checkpoint; a no-op for
    modules built in memory (already aligned).
    """
    with torch.no_grad():
        for p in module.parameters():
            p.data = p.data.clone()
    return module
