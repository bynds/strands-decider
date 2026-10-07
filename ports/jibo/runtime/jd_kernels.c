#include "jd_kernels.h"

#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define JD_NEON 1
#endif

float jd_f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
  if (exp == 0) {
    if (man == 0) bits = sign;
    else { /* subnormal: normalise */
      exp = 127 - 15 + 1;
      while (!(man & 0x400)) { man <<= 1; exp--; }
      bits = sign | (exp << 23) | ((man & 0x3FF) << 13);
    }
  } else if (exp == 31) bits = sign | 0x7F800000u | (man << 13);
  else bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

void jd_dequant_row(const jd_tensor *w, int row, float *dst) {
  int in = (int)w->dims[1];
  if (w->dtype == JD_F32) {
    memcpy(dst, (const float *)w->data + (size_t)row * in, (size_t)in * sizeof(float));
    return;
  }
  int nb = in / 32;
  if (w->dtype == JD_Q8) {
    const unsigned char *b = (const unsigned char *)w->data + (size_t)row * nb * 34;
    for (int k = 0; k < nb; k++, b += 34) {
      float s = jd_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
      const int8_t *q = (const int8_t *)(b + 2);
      for (int i = 0; i < 32; i++) dst[k * 32 + i] = s * (float)q[i];
    }
  } else { /* JD_Q4 */
    const unsigned char *b = (const unsigned char *)w->data + (size_t)row * nb * 18;
    for (int k = 0; k < nb; k++, b += 18) {
      float s = jd_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
      for (int i = 0; i < 16; i++) {
        dst[k * 32 + 2 * i] = s * (float)((int)(b[2 + i] & 15) - 8);
        dst[k * 32 + 2 * i + 1] = s * (float)((int)(b[2 + i] >> 4) - 8);
      }
    }
  }
}

static float dot(const float *a, const float *b, int n) {
#ifdef JD_NEON
  float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    s0 = vmlaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vmlaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
  }
  float32x4_t s = vaddq_f32(s0, s1);
  float32x2_t r = vadd_f32(vget_low_f32(s), vget_high_f32(s));
  float acc = vget_lane_f32(vpadd_f32(r, r), 0);
  for (; i < n; i++) acc += a[i] * b[i];
  return acc;
#else
  /* eight independent lanes, which compilers vectorise without -ffast-math */
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int i = 0;
  for (; i + 8 <= n; i += 8)
    for (int j = 0; j < 8; j++) acc[j] += a[i + j] * b[i + j];
  for (; i < n; i++) acc[0] += a[i] * b[i];
  return ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
#endif
}

void jd_matmul(float *y, int ldy, const float *x, int n, int in, const jd_tensor *w) {
  int out = (int)w->dims[0];
  /* Rows are independent; a test build with -fopenmp splits them across cores. The Jibo
   * build is single-threaded: the robot has about one core to spare. */
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    float *row = w->dtype == JD_F32 ? NULL : malloc((size_t)in * sizeof(float));
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int o = 0; o < out; o++) {
      const float *wr;
      if (w->dtype == JD_F32) wr = (const float *)w->data + (size_t)o * in;
      else { jd_dequant_row(w, o, row); wr = row; }
      for (int t = 0; t < n; t++) y[(size_t)t * ldy + o] = dot(x + (size_t)t * in, wr, in);
    }
    free(row);
  }
}
