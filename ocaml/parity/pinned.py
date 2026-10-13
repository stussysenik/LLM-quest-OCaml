"""Pinned stack for the OCaml parity references.

Every file under `ocaml/parity/` is only reproducible against an exact
transformers + torch + model-revision triple. Without the pin, a
`pip install -U transformers` silently invalidates the 652MB reference
and the resulting gate failure reads as a phantom OCaml regression.

`check_pinned()` turns that into a loud import-time failure instead.
"""

import transformers

TRANSFORMERS_VERSION = "5.19.0"
TORCH_VERSION = "2.14.1+cpu"
SAFETENSORS_VERSION = "0.8.0"
# openai-community/gpt2, the snapshot in the local HF cache.
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def banner() -> str:
    return (
        f"pinned stack: transformers=={TRANSFORMERS_VERSION} "
        f"torch=={TORCH_VERSION} safetensors=={SAFETENSORS_VERSION} "
        f"gpt2@{GPT2_REVISION}"
    )


def check_pinned() -> None:
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers {transformers.__version__} != pinned {TRANSFORMERS_VERSION}.\n"
            f"The committed references were generated with the pinned stack:\n  {banner()}\n"
            f"Recreate the venv from parity/requirements.txt and re-export every "
            f"reference file; do NOT regenerate just this one."
        )