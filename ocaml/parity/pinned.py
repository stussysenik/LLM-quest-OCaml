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
        if found != expected:
            raise RuntimeError(
                f"{label} {found} != pinned {expected}.\n"
                f"The committed references were generated with the pinned stack:\n  {banner()}\n"
                f"Recreate the venv per the header in parity/requirements.txt (torch comes\n"
                f"from the CPU index) and re-export every reference file; do NOT regenerate\n"
                f"just this one."
            )
