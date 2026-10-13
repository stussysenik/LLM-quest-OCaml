"""Export HF gpt2 *generation* reference for the OCaml end-to-end gate.

Steps 1-5 prove the forward pass and the cache match PyTorch numerically.
This proves the whole loop matches: the same token IDs, in the same order,
with the same stop point as `model.generate(do_sample=False)`.

Writes ONE safetensors file:
  - gen.prompt_ids  (I64, 1D) prompt token ids, as tokenized by HF
  - gen.new_ids     (I64, 1D) greedy continuation, plain run
  - gen.eos_ids     (I64, 1D) greedy continuation with `forced_eos_token_id`
                    — same tokens, last one replaced by EOS, which pins HF's
                    stop semantics (EOS is APPENDED, then generation stops)

Why a forced EOS: real gpt2 greedy essentially never emits <|endoftext|>
(verified over 17 prompts x 80 steps), so the stop path is unreachable
naturally. `forced_eos_token_id` is a first-class `generate` argument, so
the stop contract is tested instead of assumed.

Pinned stack — see parity/requirements.txt. The reference files are only
valid for exactly this stack; a floating `pip install -U transformers`
would silently invalidate them and look like a phantom regression.

Usage:
    python ocaml/parity/export_gen.py /data/tmp/ocaml-parity/gpt2_gen_ref.safetensors
"""

import sys

import torch
from safetensors.torch import save_file
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import pinned

pinned.check_pinned()

MAX_NEW_TOKENS = 20
PROMPT = "Hello world"


def manual_greedy(model, prompt_ids, n):
    """Greedy decode with the cache, recording the top1-top2 logit margin.

    Two jobs: cross-check `generate` (the authoritative loop) against a
    hand-rolled cached loop, and expose the margin at every step so the
    OCaml gate's exactness is known to be robust rather than lucky. A
    margin near fp32 noise would make a token mismatch meaningless.
    """
    ids = prompt_ids.clone()
    out = model(ids, use_cache=True)
    pkv = out.past_key_values
    logits = out.logits[:, -1, :]
    tokens, margins = [], []
    for _ in range(n):
        top2 = torch.topk(logits[0], 2).values
        margins.append((top2[0] - top2[1]).item())
        nxt = int(logits[0].argmax())
        tokens.append(nxt)
        step = model(torch.tensor([[nxt]]), past_key_values=pkv, use_cache=True)
        pkv = step.past_key_values
        logits = step.logits[:, -1, :]
    return tokens, margins


def main(out_path, prompt=PROMPT, max_new_tokens=MAX_NEW_TOKENS):
    model = GPT2LMHeadModel.from_pretrained("gpt2", revision=pinned.GPT2_REVISION).eval()
    tok = GPT2Tokenizer.from_pretrained("gpt2", revision=pinned.GPT2_REVISION)
    eos = tok.eos_token_id
    ids = tok(prompt, return_tensors="pt").input_ids
    print(f"{pinned.banner()}")
    print(f"prompt {prompt!r} -> ids {ids[0].tolist()}, eos={eos}")

    with torch.no_grad():
        plain = model.generate(
            ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=eos,  # unset otherwise; batch=1 never actually pads
            forced_eos_token_id=None,
        )
        # HF appends the stopping token and returns — verified, not assumed.
        stopped = model.generate(
            ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=eos,
            forced_eos_token_id=eos,
        )

    new_ids = plain[0][ids.shape[1] :].tolist()
    eos_ids = stopped[0][ids.shape[1] :].tolist()
    print(f"generate  new: {new_ids}")
    print(f"forced-eos  : {eos_ids}")
    print(f"decoded     : {tok.decode(plain[0])!r}")

    tokens, margins = manual_greedy(model, ids, max_new_tokens)
    assert tokens == new_ids, f"manual cached loop disagrees with generate: {tokens}"
    assert eos_ids[:-1] == new_ids[:-1] and eos_ids[-1] == eos, "forced-eos prefix mismatch"
    assert eos not in new_ids, "prompt unexpectedly hit EOS; forced-eos run is not a delta"
    print(f"min top1-top2 margin over {max_new_tokens} steps: {min(margins):.4f}")

    tensors = {
        "gen.prompt_ids": ids[0].cpu().clone(),
        "gen.new_ids": torch.tensor(new_ids, dtype=torch.long),
        "gen.eos_ids": torch.tensor(eos_ids, dtype=torch.long),
    }
    save_file(tensors, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/data/tmp/ocaml-parity/gpt2_gen_ref.safetensors")