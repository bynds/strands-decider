/* bench_engine: the workloads perfvm/bench.sh counts, with per-operation spans (-DJD_PROFILE).
 *
 *   bench-engine MODEL.jdw TOKENIZER.jdt REQUEST1.json REQUEST3.json [WORKLOAD]
 *
 * WORKLOAD runs just one of them (the full model's prefill takes most of an hour to count at
 * the start of the optimisation history).
 * One JSON line per workload: the clock's total over the workload and, per operation of
 * runtime/jd_prof.h, [clock, spans]; "other" is the total less the spans. The clock is the
 * instructions this process retires (perfvm/jd_icount.h) where the PMU allows, else nanoseconds.
 *
 *   prefill     REQUEST1 through jd_evaluate: render, tokenise, one forward of its prompt
 *               (state and question, 109 tokens with the fixture), the pointer head
 *   prefix_hit  after REQUEST3: copy the cached state prefix and forward one question
 *               (REQUEST3's first, 68 tokens) on it, as each question of a request does
 *   request3    REQUEST3 through jd_evaluate: the state once, then three questions on it */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../runtime/jd_engine.h"
#include "../runtime/jd_json.h"
#include "../runtime/jd_prof.h"
#include "jd_icount.h"

static uint64_t ns_clock(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

static uint64_t (*clk)(void);

static char *slurp(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *s = malloc((size_t)n + 1);
  if (s && fread(s, 1, (size_t)n, f) != (size_t)n) { free(s); s = NULL; }
  if (s) s[n] = 0;
  fclose(f);
  return s;
}

static void report(const char *workload, uint64_t total, long tokens) {
  uint64_t t[JD_P_N], c[JD_P_N], spans = 0;
  jd_prof_take(t, c);
  printf("{\"workload\": \"%s\", \"clock\": \"%s\", \"tokens\": %ld, \"total\": %llu, \"ops\": {", workload,
         clk == ns_clock ? "ns" : "instructions", tokens, (unsigned long long)total);
  for (int i = 0; i < JD_P_N; i++) {
    printf("\"%s\": [%llu, %llu], ", jd_prof_names[i], (unsigned long long)t[i], (unsigned long long)c[i]);
    spans += t[i];
  }
  printf("\"other\": [%llu, 0]}}\n", (unsigned long long)(total - spans));
}

static int evaluate(jd_engine *e, const char *path, const char *workload) {
  char err[256];
  char *json = slurp(path);
  jd_request r;
  if (!json || jd_request_parse(json, &r, err, sizeof err) != 0) { fprintf(stderr, "%s: %s\n", path, err); return -1; }
  jd_answer *a = calloc((size_t)r.n_q, sizeof(jd_answer));
  long ntok = 0;
  jd_prof_take((uint64_t[JD_P_N]){0}, (uint64_t[JD_P_N]){0});
  uint64_t t0 = clk();
  int rc = jd_evaluate(e, &r, a, &ntok, err, sizeof err);
  uint64_t t1 = clk();
  if (rc != 0) { fprintf(stderr, "evaluate: %s\n", err); return -1; }
  if (workload) report(workload, t1 - t0, ntok);
  for (int i = 0; i < r.n_q; i++) jd_answer_free(&a[i]);
  free(a); free(json);
  jd_request_free(&r);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 5 && argc != 6) {
    fprintf(stderr, "usage: bench-engine MODEL.jdw TOKENIZER.jdt REQUEST1.json REQUEST3.json [WORKLOAD]\n");
    return 2;
  }
  clk = jd_icount_open() ? jd_icount_read : ns_clock;
  jd_prof_set_clock(clk);
  jd_model m;
  jd_tokenizer tok;
  if (jd_model_load(&m, argv[1]) != 0 || jd_tok_load(&tok, argv[2]) != 0) { fprintf(stderr, "cannot load\n"); return 1; }
  jd_engine e;
  jd_engine_opts o = {0, 0, 1, 64};
  if (jd_engine_init(&e, &m, &tok, o) != 0) return 1;

  const char *only = argc == 6 ? argv[5] : NULL;
  if ((!only || !strcmp(only, "prefill")) && evaluate(&e, argv[3], "prefill")) return 1;
  if (only && !strcmp(only, "prefill")) return 0;
  if (evaluate(&e, argv[4], only && strcmp(only, "request3") ? NULL : "request3")) return 1;
  if (only && strcmp(only, "prefix_hit")) return 0;

  /* prefix_hit: e.snap still holds REQUEST3's state prefix */
  char err[256];
  char *json = slurp(argv[4]);
  jd_request r;
  if (!json || jd_request_parse(json, &r, err, sizeof err) != 0) return 1;
  char *pieces_s = jd_render_json(&r);
  jd_json *pieces = pieces_s ? jd_json_parse(pieces_s, err, sizeof err) : NULL;
  jd_tokens tq = {0};
  if (!pieces || pieces->n < 2 ||
      jd_tok_encode(&tok, pieces->items[1]->str, strlen(pieces->items[1]->str), &tq) != 0) return 1;
  jd_prof_take((uint64_t[JD_P_N]){0}, (uint64_t[JD_P_N]){0});
  uint64_t t0 = clk();
  jd_state_copy(&e.s, &e.snap, &m);
  int rc = jd_forward(&m, &e.s, tq.ids, (int)tq.n, e.hidden, o.chunk, NULL, NULL);
  uint64_t t1 = clk();
  if (rc != 0) return 1;
  report("prefix_hit", t1 - t0, (long)tq.n);
  return 0;
}
