/* jibo-decider: the decider on the CPU.
 *
 *   jibo-decider ask MODEL.jdw TOKENIZER.jdt [options] < request.json
 *
 * Reads one /v1/systemone request and prints the response, as `strands-decider serve` would
 * return it, with latency_ms. Options:
 *   --window N           context window in tokens (default: the checkpoint's max_length)
 *   --strict-window      refuse a prompt longer than the window instead of shortening it
 *   --no-prefix-cache    encode every question with its own copy of the state
 *   --chunk N            tokens per forward chunk (default 64)
 *   --raw                also print each question's unrounded probabilities, one JSON line each
 *   --model-name NAME    the "model" field of the response (default: the .jdw file name)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "jd_engine.h"

static char *read_all(FILE *f) {
  size_t cap = 1 << 16, n = 0;
  char *b = malloc(cap);
  if (!b) return NULL;
  for (;;) {
    if (n + 4096 > cap) {
      char *p = realloc(b, cap *= 2);
      if (!p) { free(b); return NULL; }
      b = p;
    }
    size_t r = fread(b + n, 1, cap - n - 1, f);
    n += r;
    if (r == 0) break;
  }
  b[n] = 0;
  return b;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
  if (argc < 4 || strcmp(argv[1], "ask") != 0) {
    fprintf(stderr, "usage: jibo-decider ask MODEL.jdw TOKENIZER.jdt [options] < request.json\n");
    return 2;
  }
  jd_engine_opts o = {0, 0, 1, 64};
  int raw = 0;
  const char *name = strrchr(argv[2], '/') ? strrchr(argv[2], '/') + 1 : argv[2];
  for (int i = 4; i < argc; i++) {
    if (!strcmp(argv[i], "--window") && i + 1 < argc) o.max_length = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--strict-window")) o.strict_window = 1;
    else if (!strcmp(argv[i], "--no-prefix-cache")) o.use_prefix_cache = 0;
    else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) o.chunk = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--raw")) raw = 1;
    else if (!strcmp(argv[i], "--model-name") && i + 1 < argc) name = argv[++i];
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  jd_model m;
  jd_tokenizer tok;
  if (jd_model_load(&m, argv[2]) != 0) { fprintf(stderr, "cannot load model %s\n", argv[2]); return 1; }
  if (jd_tok_load(&tok, argv[3]) != 0) { fprintf(stderr, "cannot load tokenizer %s\n", argv[3]); return 1; }
  jd_engine e;
  if (jd_engine_init(&e, &m, &tok, o) != 0) { fprintf(stderr, "out of memory\n"); return 1; }
  char *json = read_all(stdin);
  char err[512] = {0};
  jd_request r;
  if (!json || jd_request_parse(json, &r, err, sizeof(err)) != 0) {
    printf("{\"detail\":\"%s\"}\n", err);
    return 3;
  }
  jd_answer *ans = calloc((size_t)r.n_q, sizeof(jd_answer));
  long ntok = 0;
  double t0 = now_ms();
  if (!ans || jd_evaluate(&e, &r, ans, &ntok, err, sizeof(err)) != 0) {
    printf("{\"detail\":\"%s\"}\n", err);
    return 3;
  }
  double ms = now_ms() - t0;
  char *resp = jd_response_json(&r, ans, ntok, name, m.c.ordinal_smoothing);
  if (!resp) return 1;
  size_t L = strlen(resp);
  printf("%.*s,\"latency_ms\":%.2f}\n", (int)(L - 1), resp, ms);
  if (raw) {
    for (int i = 0; i < r.n_q; i++) {
      printf("{\"name\":\"%s\",\"labels\":[", r.q[i].name);
      for (int k = 0; k < ans[i].n; k++) printf(k ? ",\"%s\"" : "\"%s\"", ans[i].labels[k]);
      printf("],\"probs\":[");
      for (int k = 0; k < ans[i].n; k++) printf(k ? ",%.9g" : "%.9g", (double)ans[i].probs[k]);
      printf("]}\n");
    }
  }
  free(resp);
  for (int i = 0; i < r.n_q; i++) jd_answer_free(&ans[i]);
  free(ans);
  free(json);
  jd_request_free(&r);
  jd_engine_free(&e);
  jd_tok_free(&tok);
  jd_model_free(&m);
  return 0;
}
