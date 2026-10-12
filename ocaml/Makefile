LIB = lib/tensor.ml lib/config.ml lib/norm.ml lib/kv_cache.ml lib/safetensors.ml lib/attention.ml lib/block.ml lib/gpt.ml lib/sampling.ml lib/hf_loader.ml
REF ?= /data/tmp/ocaml-parity/gpt2_ref.safetensors

all: test demo

test_all: $(LIB) test/test_all.ml
	ocamlopt -I lib -o test_all $(LIB) test/test_all.ml

test_batched: $(LIB) test/test_batched.ml
	ocamlopt -I lib -o test_batched $(LIB) test/test_batched.ml

demo: $(LIB) bin/main.ml
	ocamlopt -I lib -o demo $(LIB) bin/main.ml

parity_check: $(LIB) test/parity.ml
	ocamlopt -I lib -o parity_check $(LIB) test/parity.ml

test: test_all test_batched
	./test_all && ./test_batched

demo-run: demo
	./demo

parity: parity_check
	@test -f $(REF) || (echo "reference not found: $(REF) -- run: python ocaml/parity/export_reference.py <out>" && exit 1)
	./parity_check $(REF)

clean:
	rm -f test_all test_batched demo parity_check lib/*.cmi lib/*.cmx lib/*.o test/*.cmi test/*.cmx test/*.o bin/*.cmi bin/*.cmx bin/*.o

.PHONY: all test demo-run parity clean
