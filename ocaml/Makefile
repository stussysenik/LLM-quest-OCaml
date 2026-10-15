# rope.ml precedes attention.ml: GroupedQueryAttention applies RoPE to Q/K
# internally, matching llama_attention.py's forward(x, mask, cos, sin, ...).
LIB = lib/tensor.ml lib/config.ml lib/norm.ml lib/mlp.ml lib/loss.ml lib/rope.ml lib/kv_cache.ml lib/safetensors.ml lib/attention.ml lib/block.ml lib/sampling.ml lib/gpt.ml lib/llama.ml lib/gpt2_loader.ml lib/llama_loader.ml lib/vit.ml lib/vlm.ml lib/image.ml lib/vit_loader.ml lib/gpt_repo_loader.ml lib/vlm_loader.ml
# Every reference file lives in one directory. Override it in one place
# rather than per-file: the defaults point at /data/tmp, and on the
# machine this was last verified on (Linux, 2026-10-15) that directory is
# writable and the default works as-is. It did not, on an earlier macOS
# box where SIP made /data read-only and the whole set had to live in
# /tmp. A single PARITY_DIR keeps that a one-variable change if you hit
# another machine where /data is not writable.
#
#   make ci PARITY_DIR=/tmp/ocaml-parity
#
# The individual REF_* names below are still overridable (?=) for the rare
# case of pointing one stage at a different export.
#
# No reference file is tracked in git: together they are ~670 MB, and they
# are *derived* — regenerate with ocaml/parity/requirements.txt's pinned
# venv and the export_*.py scripts, and `pinned.check_pinned()` will
# refuse to produce them under a different torch/transformers.
PARITY_DIR ?= /data/tmp/ocaml-parity
REF ?= $(PARITY_DIR)/gpt2_ref.safetensors
REF_KV ?= $(PARITY_DIR)/gpt2_kv_ref.safetensors
REF_GEN ?= $(PARITY_DIR)/gpt2_gen_ref.safetensors
REF_LLAMA ?= $(PARITY_DIR)/llama_ref.safetensors
REF_LLAMA_TIED ?= $(PARITY_DIR)/llama_tied_ref.safetensors
REF_NORM ?= $(PARITY_DIR)/norm_ref.safetensors
REF_MLP ?= $(PARITY_DIR)/mlp_ref.safetensors
REF_ROPE ?= $(PARITY_DIR)/rope_ref.safetensors
REF_ROPE_LLAMA3 ?= $(PARITY_DIR)/rope_llama3_ref.safetensors
REF_GQA ?= $(PARITY_DIR)/gqa_ref.safetensors
REF_VIT ?= $(PARITY_DIR)/vit_ref.safetensors
REF_VLM ?= $(PARITY_DIR)/vlm_ref.safetensors
REF_VLM_LOADER ?= $(PARITY_DIR)/vlm_loader_ref.safetensors
REF_IMAGE ?= $(PARITY_DIR)/image_ref.safetensors
REF_LOSS ?= $(PARITY_DIR)/loss_ref.safetensors
# Parity is parameterized by family; the dispatch lives in test/parity.ml,
# so a new family is a new loader + export script, not a new target.
FAMILY ?= gpt2

all: test demo

test_all: $(LIB) test/test_all.ml
	ocamlopt -I lib -o test_all $(LIB) test/test_all.ml

test_batched: $(LIB) test/test_batched.ml
	ocamlopt -I lib -o test_batched $(LIB) test/test_batched.ml

test_kv: $(LIB) test/test_kv.ml
	ocamlopt -I lib -o test_kv $(LIB) test/test_kv.ml

test_loader: $(LIB) test/test_loader.ml
	ocamlopt -I lib -o test_loader $(LIB) test/test_loader.ml

test_norm: $(LIB) test/test_norm.ml
	ocamlopt -I lib -o test_norm $(LIB) test/test_norm.ml

test_mlp: $(LIB) test/test_mlp.ml
	ocamlopt -I lib -o test_mlp $(LIB) test/test_mlp.ml

test_rope: $(LIB) test/test_rope.ml
	ocamlopt -I lib -o test_rope $(LIB) test/test_rope.ml

test_gqa: $(LIB) test/test_gqa.ml
	ocamlopt -I lib -o test_gqa $(LIB) test/test_gqa.ml

test_vit: $(LIB) test/test_vit.ml
	ocamlopt -I lib -o test_vit $(LIB) test/test_vit.ml

test_vlm: $(LIB) test/test_vlm.ml
	ocamlopt -I lib -o test_vlm $(LIB) test/test_vlm.ml

test_loader_vlm: $(LIB) test/test_loader_vlm.ml
	ocamlopt -I lib -o test_loader_vlm $(LIB) test/test_loader_vlm.ml

test_image: $(LIB) test/test_image.ml
	ocamlopt -I lib -o test_image $(LIB) test/test_image.ml

test_loss: $(LIB) test/test_loss.ml
	ocamlopt -I lib -o test_loss $(LIB) test/test_loss.ml

demo: $(LIB) bin/main.ml
	ocamlopt -I lib -o demo $(LIB) bin/main.ml

parity_check: $(LIB) test/parity.ml
	ocamlopt -I lib -o parity_check $(LIB) test/parity.ml

parity_kv: $(LIB) test/parity_kv.ml
	ocamlopt -I lib -o parity_kv $(LIB) test/parity_kv.ml

parity_gen: $(LIB) test/parity_gen.ml
	ocamlopt -I lib -o parity_gen $(LIB) test/parity_gen.ml

test: test_all test_batched test_kv test_loader test_norm test_mlp test_rope test_gqa test_vit test_vlm
	./test_all && ./test_batched && ./test_kv && ./test_loader $(REF_LLAMA) $(REF_LLAMA_TIED)
	@test -f $(REF_NORM) || (echo "norm reference not found: $(REF_NORM) -- run: python ocaml/parity/export_norm.py $(REF_NORM)" && exit 1)
	./test_norm $(REF_NORM)
	@test -f $(REF_MLP) || (echo "mlp reference not found: $(REF_MLP) -- run: python ocaml/parity/export_mlp.py $(REF_MLP)" && exit 1)
	./test_mlp $(REF_MLP)
	@test -f $(REF_ROPE) || (echo "rope reference not found: $(REF_ROPE) -- run: python ocaml/parity/export_rope.py $(REF_ROPE)" && exit 1)
	./test_rope $(REF_ROPE) $(REF_ROPE_LLAMA3)
	@test -f $(REF_GQA) || (echo "gqa reference not found: $(REF_GQA) -- run: python ocaml/parity/export_gqa.py $(REF_GQA)" && exit 1)
	./test_gqa $(REF_GQA)
	@test -f $(REF_VIT) || (echo "vit reference not found: $(REF_VIT) -- run: python ocaml/parity/export_vit.py $(REF_VIT)" && exit 1)
	./test_vit $(REF_VIT)
	@test -f $(REF_VLM) || (echo "vlm reference not found: $(REF_VLM) -- run: python ocaml/parity/export_vlm.py $(REF_VLM)" && exit 1)
	./test_vlm $(REF_VLM)
	@test -f $(REF_VLM_LOADER) || (echo "vlm loader reference not found: $(REF_VLM_LOADER) -- run: python ocaml/parity/export_loader.py $(REF_VLM_LOADER)" && exit 1)
	./test_loader_vlm $(REF_VLM_LOADER)
	@test -f $(REF_IMAGE) || (echo "image reference not found: $(REF_IMAGE) -- run: python ocaml/parity/export_image.py $(REF_IMAGE)" && exit 1)
	./test_image $(REF_IMAGE)
	@test -f $(REF_LOSS) || (echo "loss reference not found: $(REF_LOSS) -- run: python ocaml/parity/export_loss.py $(REF_LOSS)" && exit 1)
	./test_loss $(REF_LOSS)

demo-run: demo
	./demo

parity: parity_check
	@test -f $(REF) || (echo "reference not found: $(REF) -- run: python ocaml/parity/export_reference.py <out>" && exit 1)
	./parity_check --family $(FAMILY) $(REF)

parity-kv: parity_kv
	@test -f $(REF) || (echo "weights not found: $(REF)" && exit 1)
	@test -f $(REF_KV) || (echo "kv reference not found: $(REF_KV) -- run: python ocaml/parity/export_kv.py <out>" && exit 1)
	./parity_kv $(REF) $(REF_KV)

parity-gen: parity_gen
	@test -f $(REF) || (echo "weights not found: $(REF)" && exit 1)
	@test -f $(REF_GEN) || (echo "gen reference not found: $(REF_GEN) -- run: python ocaml/parity/export_gen.py <out>" && exit 1)
	./parity_gen $(REF) $(REF_GEN)


# E1: --family llama must run the *real* forward now that D3/D4/D5 are
# landed. Before step E it had to fail AT the step-D boundary, which was a
# gate that the loader worked and the kernels did not. That gate is spent;
# what replaces it checks the thing that actually matters — that the
# llama path is not silently falling back to GPT-2 shapes or a stub, and
# that its layout gate ran. A bare exit-0 would pass just as well for a
# harness that quietly did nothing.
parity-llama-forward: parity_check
	@test -f $(REF_LLAMA) || (echo "llama reference not found: $(REF_LLAMA) -- run: python ocaml/parity/export_reference.py --family llama $(REF_LLAMA)" && exit 1)
	@out=$$(./parity_check --family llama $(REF_LLAMA) 2>&1); status=$$?; \
	if [ $$status -ne 0 ]; then echo "E1 FAIL: --family llama exited $$status"; echo "$$out"; exit 1; fi; \
	echo "$$out" | grep -q "layout gate: llama linear weights are \[out, in\]" || { echo "E1 FAIL: llama layout gate did not run"; echo "$$out"; exit 1; }; \
	echo "$$out" | grep -q "^logits .*OK$$" || { echo "E1 FAIL: no logits stage reached"; echo "$$out"; exit 1; }; \
	echo "E1: --family llama runs the step-E forward (layout gate + logits stage)"

ci: test parity parity-kv parity-gen parity-llama-forward

clean:
	rm -f test_all test_batched test_kv test_loader test_norm test_mlp test_rope test_gqa test_vit test_vlm test_loader_vlm test_image test_loss demo parity_check parity_kv parity_gen lib/*.cmi lib/*.cmx lib/*.o test/*.cmi test/*.cmx test/*.o bin/*.cmi bin/*.cmx bin/*.o

.PHONY: all test demo-run parity parity-kv parity-gen parity-llama-forward ci clean
