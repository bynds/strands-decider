/* bench-op: one real operation of the model, on the CPU and on the GPU (Phase 6 of docs/jibo-port.md).
 *
 *   bench-op MODEL.jdw TENSOR [--tokens N] [--reps R] [--tt K] [--rows-per-dispatch R]
 *            [--per-fence F] [--cpu-only] [--libgl PATH]
 *
 * TENSOR is a matrix of the .jdw, e.g. l0.up (an MLP projection). Activations are N x in
 * deterministic pseudo-random values. The CPU side is jd_matmul, the GPU side jd_gl_matmul, on the
 * same weights; the GPU result is checked against the CPU's. Prints one JSON line with medians over
 * R repetitions: CPU time; GPU weight upload (once), activation upload, dispatches until the fence
 * signals, readback, their host total, and the GPU time the timer queries report; and the largest
 * difference relative to the largest output. Run it within the agreed deployment scope; it touches
 * nothing but its own context.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../runtime/jd_gl.h"
#include "../runtime/jd_kernels.h"
#include "../runtime/jd_model.h"

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static int cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}
static double median(double *v, int n) {
  qsort(v, (size_t)n, sizeof(double), cmpd);
  return v[n / 2];
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: bench-op MODEL.jdw TENSOR [--tokens N] [--reps R] [--tt K] "
                    "[--rows-per-dispatch R] [--per-fence F] [--cpu-only] [--libgl PATH]\n");
    return 2;
  }
  int n = 128, reps = 5, cpu_only = 0;
  jd_gl_tiling tl = {8, 0, 1};
  const char *libgl = NULL;
  for (int i = 3; i < argc; i++) {
    if (!strcmp(argv[i], "--tokens") && i + 1 < argc) n = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--tt") && i + 1 < argc) tl.tokens_per_thread = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--rows-per-dispatch") && i + 1 < argc) tl.rows_per_dispatch = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--per-fence") && i + 1 < argc) tl.dispatches_per_fence = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--cpu-only")) cpu_only = 1;
    else if (!strcmp(argv[i], "--libgl") && i + 1 < argc) libgl = argv[++i];
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  if (reps < 1) reps = 1;
  jd_model m;
  jd_tensor w;
  if (jd_model_load(&m, argv[1]) != 0 || jd_model_find(&m, argv[2], &w) != 0 || w.ndim != 2) {
    fprintf(stderr, "cannot load %s or find matrix %s\n", argv[1], argv[2]);
    return 1;
  }
  int rows = (int)w.dims[0], in = (int)w.dims[1];
  float *x = malloc((size_t)n * in * sizeof(float)), *yc = malloc((size_t)n * rows * sizeof(float)),
        *yg = malloc((size_t)n * rows * sizeof(float));
  double *t = malloc(sizeof(double) * (size_t)reps * 6);
  if (!x || !yc || !yg || !t) return 1;
  uint32_t s = 12345;
  for (long i = 0; i < (long)n * in; i++) {
    s = s * 1664525u + 1013904223u;
    x[i] = (float)((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
  }
  for (int r = 0; r < reps; r++) {
    double t0 = now_ms();
    jd_matmul(yc, rows, x, n, in, &w);
    t[r] = now_ms() - t0;
  }
  double cpu = median(t, reps);
  double gflop = 2.0 * n * (double)rows * in / 1e9;
  printf("{\"tensor\":\"%s\",\"dtype\":%d,\"rows\":%d,\"cols\":%d,\"tokens\":%d,\"gflop\":%.4f,\"cpu_ms\":%.3f,"
         "\"cpu_gflops\":%.3f",
         argv[2], w.dtype, rows, in, n, gflop, cpu, gflop / (cpu / 1e3));
  if (cpu_only) { printf("}\n"); return 0; }

  char err[256];
  jd_gl *g = jd_gl_create(libgl, err, sizeof(err));
  if (!g) { printf(",\"gpu_error\":\"%s\"}\n", err); return 1; }
  jd_gl_matrix gw;
  double t0 = now_ms();
  if (jd_gl_upload_matrix(g, &w, &gw) != 0) { printf(",\"gpu_error\":\"upload\"}\n"); return 1; }
  unsigned int yb = jd_gl_buffer(g, NULL, (long)n * rows * 4);
  jd_gl_read(g, yb, yg, 4); /* forces the upload to complete before timing */
  double upload_w = now_ms() - t0;
  double *tx = t + reps, *td = t + 2 * reps, *tr = t + 3 * reps, *tt = t + 4 * reps, *tg = t + 5 * reps;
  int nd = 0;
  for (int r = 0; r < reps; r++) {
    double a = now_ms();
    unsigned int xb = jd_gl_buffer(g, x, (long)n * in * 4);
    double b = now_ms();
    if (jd_gl_matmul(g, &gw, xb, n, yb, rows, tl, &tg[r], &nd) != 0) { printf(",\"gpu_error\":\"matmul\"}\n"); return 1; }
    double c = now_ms();
    if (jd_gl_read(g, yb, yg, (long)n * rows * 4) != 0) { printf(",\"gpu_error\":\"read\"}\n"); return 1; }
    double d = now_ms();
    jd_gl_free_buffer(g, xb);
    tx[r] = b - a; td[r] = c - b; tr[r] = d - c; tt[r] = d - a;
  }
  double maxabs = 0, maxdiff = 0;
  for (long i = 0; i < (long)n * rows; i++) {
    double v = yc[i] < 0 ? -yc[i] : yc[i], dlt = yc[i] - yg[i];
    if (dlt < 0) dlt = -dlt;
    if (v > maxabs) maxabs = v;
    if (dlt > maxdiff) maxdiff = dlt;
  }
  printf(",\"renderer\":\"%s\",\"tt\":%d,\"rows_per_dispatch\":%d,\"per_fence\":%d,\"dispatches\":%d,"
         "\"gpu_upload_weights_ms\":%.3f,\"gpu_upload_x_ms\":%.3f,\"gpu_dispatch_wait_ms\":%.3f,"
         "\"gpu_readback_ms\":%.3f,\"gpu_host_total_ms\":%.3f,\"gpu_timer_ms\":%.3f,"
         "\"gpu_gflops_host\":%.3f,\"max_rel_diff\":%.3g}\n",
         jd_gl_renderer(g), tl.tokens_per_thread, tl.rows_per_dispatch, tl.dispatches_per_fence, nd, upload_w,
         median(tx, reps), median(td, reps), median(tr, reps), median(tt, reps), median(tg, reps),
         gflop / (median(tt, reps) / 1e3), maxdiff / (maxabs > 0 ? maxabs : 1));
  jd_gl_free_matrix(g, &gw);
  jd_gl_free_buffer(g, yb);
  jd_gl_destroy(g);
  jd_model_free(&m);
  return maxdiff / (maxabs > 0 ? maxabs : 1) < 1e-4 ? 0 : 3;
}
