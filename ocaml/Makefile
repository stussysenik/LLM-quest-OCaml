LIB = lib/tensor.ml lib/config.ml lib/norm.ml lib/kv_cache.ml lib/safetensors.ml lib/attention.ml lib/block.ml lib/sampling.ml lib/gpt.ml lib/hf_loader.ml
REF ?= /data/tmp/ocaml-parity/gpt2_ref.safetensors
REF_KV ?= /data/tmp/ocaml-parity/gpt2_kv_ref.safetensors

all: test demo

test_all: $(LIB) test/test_all.ml
	ocamlopt -I lib -o test_all $(LIB) test/test_all.ml

test_batched: $(LIB) test/test_batched.ml
	ocamlopt -I lib -o test_batched $(LIB) test/test_batched.ml

test_kv: $(LIB) test/test_kv.ml
	ocamlopt -I lib -o test_kv $(LIB) test/test_kv.ml

demo: $(LIB) bin/main.ml
	ocamlopt -I lib -o demo $(LIB) bin/main.ml

parity_check: $(LIB) test/parity.ml
	ocamlopt -I lib -o parity_check $(LIB) test/parity.ml

parity_kv: $(LIB) test/parity_kv.ml
	ocamlopt -I lib -o parity_kv $(LIB) test/parity_kv.ml

test: test_all test_batched test_kv
	./test_all && ./test_batched && ./test_kv

demo-run: demo
	./demo

parity: parity_check
	@test -f $(REF) || (echo "reference not found: $(REF) -- run: python ocaml/parity/export_reference.py <out>" && exit 1)
	./parity_check $(REF)

parity-kv: parity_kv
	@test -f $(REF) || (echo "weights not found: $(REF)" && exit 1)
	@test -f $(REF_KV) || (echo "kv reference not found: $(REF_KV) -- run: python ocaml/parity/export_kv.py <out>" && exit 1)
	./parity_kv $(REF) $(REF_KV)

ci: test parity parity-kv

clean:
	rm -f test_all test_batched test_kv demo parity_check parity_kv lib/*.cmi lib/*.cmx lib/*.o test/*.cmi test/*.cmx test/*.o bin/*.cmi bin/*.cmx bin/*.o

.PHONY: all test demo-run parity parity-kv ci clean
