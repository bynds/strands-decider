/* Test helper: run token ids through the C torso and write every layer's output.
 *
 *   fwd_dump MODEL.jdw IDS.bin OUT.bin [CHUNK] [SPLIT]
 *
 * IDS.bin is int32 token ids. OUT.bin is float32: n_layers + 1 blocks of n x hidden, each
 * layer's output and then the final normed hidden state. With SPLIT > 0 the first SPLIT
 * tokens are run, the state is copied into a second state, and the rest continue from the
 * copy: the path the shared prefix takes. */
#include <stdio.h>
#include <stdlib.h>

#include "../runtime/jd_model.h"

typedef struct {
  float *buf;
  int n, hidden;
} dump_ctx;

static void on_layer(void *vctx, int layer, int t0, int n, const float *h, int hidden) {
  dump_ctx *c = vctx;
  for (int t = 0; t < n; t++)
    for (int i = 0; i < hidden; i++)
      c->buf[((size_t)layer * c->n + t0 + t) * hidden + i] = h[(size_t)t * hidden + i];
}

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: fwd_dump MODEL IDS OUT [CHUNK] [SPLIT]\n"); return 2; }
  int chunk = argc > 4 ? atoi(argv[4]) : 64, split = argc > 5 ? atoi(argv[5]) : 0;
  jd_model m;
  if (jd_model_load(&m, argv[1]) != 0) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
  FILE *f = fopen(argv[2], "rb");
  if (!f) return 1;
  fseek(f, 0, SEEK_END);
  long bytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  int n = (int)(bytes / 4);
  int32_t *ids = malloc((size_t)n * 4);
  if (!ids || fread(ids, 4, (size_t)n, f) != (size_t)n) return 1;
  fclose(f);
  int H = m.c.hidden, L = m.c.n_layers;
  dump_ctx ctx = {calloc((size_t)(L + 1) * n * H, sizeof(float)), n, H};
  float *final = ctx.buf + (size_t)L * n * H;
  jd_state s, s2;
  if (!ctx.buf || jd_state_init(&s, &m, n) != 0 || jd_state_init(&s2, &m, n) != 0) return 1;
  if (split > 0 && split < n) {
    if (jd_forward(&m, &s, ids, split, final, chunk, on_layer, &ctx) != 0) return 1;
    jd_state_copy(&s2, &s, &m);
    jd_state_reset(&s, &m); /* the original must not be needed again */
    if (jd_forward(&m, &s2, ids + split, n - split, final + (size_t)split * H, chunk, on_layer, &ctx) != 0)
      return 1;
  } else if (jd_forward(&m, &s, ids, n, final, chunk, on_layer, &ctx) != 0) {
    return 1;
  }
  f = fopen(argv[3], "wb");
  if (!f || fwrite(ctx.buf, sizeof(float), (size_t)(L + 1) * n * H, f) != (size_t)(L + 1) * n * H) return 1;
  fclose(f);
  jd_state_free(&s, &m);
  jd_state_free(&s2, &m);
  jd_model_free(&m);
  return 0;
}
