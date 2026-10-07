/* The desktop-GL compute backend: one context, owned by the thread that created it. */
#ifndef JD_GL_H
#define JD_GL_H

#include <stdint.h>

#include "jd_model.h"

typedef struct jd_gl jd_gl;

typedef struct {
  int dtype;           /* JD_F32, JD_Q8 or JD_Q4 */
  int rows, cols;
  unsigned int buf_w;  /* f32: floats; q8/q4: packed quants, 4 uints per block of 32 (q4), 8 (q8) */
  unsigned int buf_s;  /* q8/q4: scales, two fp16 per uint */
} jd_gl_matrix;

typedef struct {
  int tokens_per_thread; /* outputs per invocation along the token axis (registers) */
  int rows_per_dispatch; /* bounds one dispatch's work so the renderer is not held up */
  int dispatches_per_fence; /* how many dispatches are queued before waiting */
} jd_gl_tiling;

/* Create a GL 4.3 core context on a 1x1 pbuffer of $DISPLAY. libgl: NULL for libGL.so.1.
 * Returns NULL and fills err on failure. */
jd_gl *jd_gl_create(const char *libgl, char *err, int err_len);
void jd_gl_destroy(jd_gl *g);
const char *jd_gl_renderer(jd_gl *g);

int jd_gl_upload_matrix(jd_gl *g, const jd_tensor *w, jd_gl_matrix *out);
void jd_gl_free_matrix(jd_gl *g, jd_gl_matrix *m);

unsigned int jd_gl_buffer(jd_gl *g, const void *data, long bytes); /* a storage buffer */
int jd_gl_read(jd_gl *g, unsigned int buf, void *dst, long bytes);  /* barrier, then read */
void jd_gl_free_buffer(jd_gl *g, unsigned int buf);

/* y[t * ldy + o] = sum_i x[t * in + i] * W[o, i], on the GPU, x and y being buffers. Waits for
 * completion with blocking fences. *gpu_ms receives the summed GL_TIME_ELAPSED of the dispatches,
 * or -1 if the timer query is unavailable. Returns 0, or -1 on a GL error. */
int jd_gl_matmul(jd_gl *g, const jd_gl_matrix *w, unsigned int x, int n, unsigned int y, int ldy,
                 jd_gl_tiling tiling, double *gpu_ms, int *dispatches);

#endif
