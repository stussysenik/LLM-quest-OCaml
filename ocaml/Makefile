LIB = lib/tensor.ml lib/config.ml lib/norm.ml lib/mlp.ml lib/kv_cache.ml lib/safetensors.ml lib/attention.ml lib/block.ml lib/sampling.ml lib/gpt.ml lib/llama.ml lib/gpt2_loader.ml lib/llama_loader.ml
REF ?= /data/tmp/ocaml-parity/gpt2_ref.safetensors
REF_KV ?= /data/tmp/ocaml-parity/gpt2_kv_ref.safetensors
REF_GEN ?= /data/tmp/ocaml-parity/gpt2_gen_ref.safetensors
REF_LLAMA ?= /data/tmp/ocaml-parity/llama_ref.safetensors
REF_LLAMA_TIED ?= /data/tmp/ocaml-parity/llama_tied_ref.safetensors
REF_NORM ?= /data/tmp/ocaml-parity/norm_ref.safetensors
REF_MLP ?= /data/tmp/ocaml-parity/mlp_ref.safetensors
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

demo: $(LIB) bin/main.ml
	ocamlopt -I lib -o demo $(LIB) bin/main.ml

parity_check: $(LIB) test/parity.ml
	ocamlopt -I lib -o parity_check $(LIB) test/parity.ml

parity_kv: $(LIB) test/parity_kv.ml
	ocamlopt -I lib -o parity_kv $(LIB) test/parity_kv.ml

parity_gen: $(LIB) test/parity_gen.ml
	ocamlopt -I lib -o parity_gen $(LIB) test/parity_gen.ml

test: test_all test_batched test_kv test_loader test_norm test_mlp
	./test_all && ./test_batched && ./test_kv && ./test_loader $(REF_LLAMA) $(REF_LLAMA_TIED)
	@test -f $(REF_NORM) || (echo "norm reference not found: $(REF_NORM) -- run: python ocaml/parity/export_norm.py $(REF_NORM)" && exit 1)
	./test_norm $(REF_NORM)
	@test -f $(REF_MLP) || (echo "mlp reference not found: $(REF_MLP) -- run: python ocaml/parity/export_mlp.py $(REF_MLP)" && exit 1)
	./test_mlp $(REF_MLP)

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

# C5: --family llama must fail AT the step D boundary — weights load,
# kernels do not exist. A gate that only checks the happy path cannot
# tell "the loader is done and the kernels are not" from "nobody wrote
# it": both look like a non-zero exit. So assert both halves — non-zero,
# and the message names step D.
parity-llama-boundary: parity_check
	@test -f $(REF_LLAMA) || (echo "llama reference not found: $(REF_LLAMA) -- run: python ocaml/parity/export_reference.py --family llama $(REF_LLAMA)" && exit 1)
	@out=$$(./parity_check --family llama $(REF_LLAMA) 2>&1); status=$$?; \
	if [ $$status -eq 0 ]; then echo "C5 FAIL: --family llama succeeded; it must stop at ROADMAP step D"; exit 1; fi; \
	if ! echo "$$out" | grep -q "ROADMAP step D"; then echo "C5 FAIL: --family llama failed, but not at the step D boundary:"; echo "$$out"; exit 1; fi; \
	echo "C5 boundary: --family llama loads weights, stops at ROADMAP step D"

ci: test parity parity-kv parity-gen parity-llama-boundary

clean:
	rm -f test_all test_batched test_kv test_loader test_norm test_mlp demo parity_check parity_kv parity_gen lib/*.cmi lib/*.cmx lib/*.o test/*.cmi test/*.cmx test/*.o bin/*.cmi bin/*.cmx bin/*.o

.PHONY: all test demo-run parity parity-kv parity-gen parity-llama-boundary ci clean
