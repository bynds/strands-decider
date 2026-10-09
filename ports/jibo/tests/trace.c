/* trace: the bit-parity gate's trace (perfvm/gate.sh). Writes every number a change to the
 * runtime could move, as raw bytes, so two builds compare with cmp(1):
 *
 *   trace MODEL.jdw TOKENIZER.jdt REQUEST1.json REQUEST3.json OUT
 *
 *   ids          the token ids of REQUEST1's prompt (its state, then its first question)
 *   layer<i>     every layer's output hidden states over that prompt, run in chunks of 64
 *   final        the final normed hidden states
 *   step<j>      8 one-token forwards continuing from that state (the matrix-vector path)
 *   probs1       the answers of REQUEST1 through jd_evaluate
 *   probs3       the answers of REQUEST3, three questions sharing the state's prefix
 *   probs3_flat  the same without the prefix cache
 *
 * Each section is a 16-byte name, a u32 byte count and the bytes. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../runtime/jd_engine.h"
#include "../runtime/jd_json.h"
#include "../runtime/jd_model.h"
#include "../runtime/jd_tokenizer.h"

static FILE *out;

static void section(const char *name, const void *data, size_t bytes) {
  char n[16] = {0};
  strncpy(n, name, sizeof n - 1);
  uint32_t b = (uint32_t)bytes;
  fwrite(n, 1, 16, out);
  fwrite(&b, 4, 1, out);
  fwrite(data, 1, bytes, out);
}

static void dump(void *ctx, int layer, int t0, int n, const float *h, int hidden) {
  (void)ctx;
  char name[16];
  snprintf(name, sizeof name, "layer%d@%d", layer, t0);
  section(name, h, (size_t)n * hidden * sizeof(float));
}

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

static int answers(jd_engine *e, const char *path, const char *name) {
  char err[256];
  char *json = slurp(path);
  jd_request r;
  if (!json || jd_request_parse(json, &r, err, sizeof err) != 0) { fprintf(stderr, "%s: %s\n", path, err); return -1; }
  jd_answer *a = calloc((size_t)r.n_q, sizeof(jd_answer));
  long ntok = 0;
  if (!a || jd_evaluate(e, &r, a, &ntok, err, sizeof err) != 0) { fprintf(stderr, "evaluate: %s\n", err); return -1; }
  size_t total = 0;
  for (int i = 0; i < r.n_q; i++) total += (size_t)a[i].n;
  float *p = malloc((total + 1) * sizeof(float)), *q = p;
  for (int i = 0; i < r.n_q; i++) {
    memcpy(q, a[i].probs, (size_t)a[i].n * sizeof(float));
    q += a[i].n;
    jd_answer_free(&a[i]);
  }
  *q = (float)ntok;
  section(name, p, (total + 1) * sizeof(float));
  free(p); free(a); free(json);
  jd_request_free(&r);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: trace MODEL.jdw TOKENIZER.jdt REQUEST1.json REQUEST3.json OUT\n");
    return 2;
  }
  jd_model m;
  jd_tokenizer tok;
  if (jd_model_load(&m, argv[1]) != 0 || jd_tok_load(&tok, argv[2]) != 0) { fprintf(stderr, "cannot load\n"); return 1; }
  out = fopen(argv[5], "wb");
  if (!out) return 1;
  int H = m.c.hidden;

  /* REQUEST1's prompt as the engine builds it without truncation: state block, then question */
  char err[256];
  char *json = slurp(argv[3]);
  jd_request r;
  if (!json || jd_request_parse(json, &r, err, sizeof err) != 0) { fprintf(stderr, "%s\n", err); return 1; }
  char *pieces_s = jd_render_json(&r);
  jd_json *pieces = pieces_s ? jd_json_parse(pieces_s, err, sizeof err) : NULL;
  if (!pieces || pieces->type != JD_JARRAY || pieces->n < 2) { fprintf(stderr, "render failed\n"); return 1; }
  jd_tokens ts = {0}, tq = {0};
  if (jd_tok_encode(&tok, pieces->items[0]->str, strlen(pieces->items[0]->str), &ts) != 0 ||
      jd_tok_encode(&tok, pieces->items[1]->str, strlen(pieces->items[1]->str), &tq) != 0) return 1;
  int n = (int)(ts.n + tq.n);
  int32_t *ids = malloc((size_t)(n + 8) * sizeof(int32_t));
  for (size_t i = 0; i < ts.n; i++) ids[i] = ts.ids[i];
  for (size_t i = 0; i < tq.n; i++) ids[ts.n + i] = tq.ids[i];
  section("ids", ids, (size_t)n * sizeof(int32_t));

  jd_state s;
  float *hid = malloc((size_t)(n + 8) * H * sizeof(float));
  if (!hid || jd_state_init(&s, &m, n + 8) != 0) return 1;
  if (jd_forward(&m, &s, ids, n, hid, 64, dump, NULL) != 0) return 1;
  section("final", hid, (size_t)n * H * sizeof(float));
  for (int j = 0; j < 8; j++) {
    int32_t id = ids[(j * 37) % n];
    char name[16];
    snprintf(name, sizeof name, "step%d", j);
    if (jd_forward(&m, &s, &id, 1, hid, 64, NULL, NULL) != 0) return 1;
    section(name, hid, (size_t)H * sizeof(float));
  }

  jd_engine e;
  jd_engine_opts o = {0, 0, 1, 64};
  if (jd_engine_init(&e, &m, &tok, o) != 0) return 1;
  if (answers(&e, argv[3], "probs1") || answers(&e, argv[4], "probs3")) return 1;
  e.o.use_prefix_cache = 0;
  if (answers(&e, argv[4], "probs3_flat")) return 1;
  fclose(out);
  printf("trace: %d prompt tokens, %d layers -> %s\n", n, m.c.n_layers, argv[5]);
  return 0;
}
