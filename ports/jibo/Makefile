# The C runtime. `make` builds for this machine; scripts/build-jibo.sh cross-builds for Jibo.
CC ?= cc
CFLAGS ?= -O2
CFLAGS += -std=c99 -Wall -Wextra -D_POSIX_C_SOURCE=200809L
LDLIBS += -lm
OUT ?= build

RUNTIME = runtime/jd_unicode.c runtime/jd_tokenizer.c runtime/jd_kernels.c runtime/jd_model.c \
          runtime/jd_engine.c runtime/jd_json.c
HEADERS = $(wildcard runtime/*.h)

all: $(OUT)/jibo-decider $(OUT)/tok_dump $(OUT)/fwd_dump $(OUT)/bench-op $(OUT)/jibo-gl-probe

$(OUT)/jibo-decider: runtime/main.c $(RUNTIME) $(HEADERS)
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) -o $@ runtime/main.c $(RUNTIME) $(LDFLAGS) $(LDLIBS)

$(OUT)/tok_dump: tests/tok_dump.c runtime/jd_unicode.c runtime/jd_tokenizer.c $(HEADERS)
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) -o $@ tests/tok_dump.c runtime/jd_unicode.c runtime/jd_tokenizer.c $(LDFLAGS) $(LDLIBS)

$(OUT)/fwd_dump: tests/fwd_dump.c runtime/jd_model.c runtime/jd_kernels.c $(HEADERS)
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) -o $@ tests/fwd_dump.c runtime/jd_model.c runtime/jd_kernels.c $(LDFLAGS) $(LDLIBS)

$(OUT)/bench-op: tools/bench_op.c runtime/jd_gl.c runtime/jd_model.c runtime/jd_kernels.c $(HEADERS)
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) -o $@ tools/bench_op.c runtime/jd_gl.c runtime/jd_model.c runtime/jd_kernels.c $(LDFLAGS) -ldl $(LDLIBS)

$(OUT)/jibo-gl-probe: tools/jibo-gl-probe.c
	@mkdir -p $(OUT)
	$(CC) $(CFLAGS) -o $@ tools/jibo-gl-probe.c $(LDFLAGS) -ldl

runtime/unicode_tables.h: tools/gen_unicode_tables.py
	python tools/gen_unicode_tables.py > $@

clean:
	rm -rf $(OUT)

.PHONY: all clean
