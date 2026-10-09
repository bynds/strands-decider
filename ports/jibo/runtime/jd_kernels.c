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
  /* Eight independent lanes, which compilers vectorise without -ffast-math. Named, not an
   * array: GCC keeps an accumulator array on the stack on 32-bit ARM, a load and a store per
   * multiply-add. */
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  int i = 0;
#if defined(__arm__) && defined(__ARM_FP)
  /* VFP: two 8-register loads with post-increment and eight non-fused vmla per 8 inputs, 12
   * instructions where GCC emits 29. Each lane adds its products in the same order as the C
   * below, so the result is the same to the bit. s16-s31 are callee-saved: as clobbers, GCC
   * saves them once per call. */
  if (n >= 8) {
    const float *pa = a, *pb = b;
    int k = n / 8;
    __asm__ volatile(
        "1:\n\t"
        "vldmia %[pa]!, {s16-s23}\n\t"
        "vldmia %[pb]!, {s24-s31}\n\t"
        "vmla.f32 %[s0], s16, s24\n\t"
        "vmla.f32 %[s1], s17, s25\n\t"
        "vmla.f32 %[s2], s18, s26\n\t"
        "vmla.f32 %[s3], s19, s27\n\t"
        "vmla.f32 %[s4], s20, s28\n\t"
        "vmla.f32 %[s5], s21, s29\n\t"
        "vmla.f32 %[s6], s22, s30\n\t"
        "vmla.f32 %[s7], s23, s31\n\t"
        "subs %[k], %[k], #1\n\t"
        "bne 1b"
        : [pa] "+r"(pa), [pb] "+r"(pb), [k] "+r"(k), [s0] "+t"(s0), [s1] "+t"(s1), [s2] "+t"(s2),
          [s3] "+t"(s3), [s4] "+t"(s4), [s5] "+t"(s5), [s6] "+t"(s6), [s7] "+t"(s7)
        :
        : "cc", "memory", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26",
          "s27", "s28", "s29", "s30", "s31");
    i = n & ~7;
  }
#endif
  for (; i + 8 <= n; i += 8) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
    s4 += a[i + 4] * b[i + 4];
    s5 += a[i + 5] * b[i + 5];
    s6 += a[i + 6] * b[i + 6];
    s7 += a[i + 7] * b[i + 7];
  }
  for (; i < n; i++) s0 += a[i] * b[i];
  return ((s0 + s4) + (s1 + s5)) + ((s2 + s6) + (s3 + s7));
#endif
}

#if defined(__arm__) && defined(__ARM_FP) && !defined(JD_NEON)
#define JD_VFP_DOT2 1
/* Two tokens against one weight row: what dot computes for each, with the weights loaded once
 * for both. Per 8 inputs, three 8-register loads and sixteen vmla; each token keeps its eight
 * lanes and their order. The lanes live in s0-s15 (token 0 in s0-s7), loaded from and stored to
 * acc around the loop; the inputs pass through s16-s31. */
static void dot2(const float *x0, const float *x1, const float *w, int n, float *y0, float *y1) {
  float acc[16] = {0};
  int i = 0;
  if (n >= 8) {
    const float *p0 = x0, *p1 = x1, *pw = w;
    int k = n / 8;
    __asm__ volatile(
        "vldmia %[acc], {s0-s15}\n\t"
        "1:\n\t"
        "vldmia %[pw]!, {s16-s23}\n\t"
        "vldmia %[p0]!, {s24-s31}\n\t"
        "vmla.f32 s0, s24, s16\n\t"
        "vmla.f32 s1, s25, s17\n\t"
        "vmla.f32 s2, s26, s18\n\t"
        "vmla.f32 s3, s27, s19\n\t"
        "vmla.f32 s4, s28, s20\n\t"
        "vmla.f32 s5, s29, s21\n\t"
        "vmla.f32 s6, s30, s22\n\t"
        "vmla.f32 s7, s31, s23\n\t"
        "vldmia %[p1]!, {s24-s31}\n\t"
        "vmla.f32 s8, s24, s16\n\t"
        "vmla.f32 s9, s25, s17\n\t"
        "vmla.f32 s10, s26, s18\n\t"
        "vmla.f32 s11, s27, s19\n\t"
        "vmla.f32 s12, s28, s20\n\t"
        "vmla.f32 s13, s29, s21\n\t"
        "vmla.f32 s14, s30, s22\n\t"
        "vmla.f32 s15, s31, s23\n\t"
        "subs %[k], %[k], #1\n\t"
        "bne 1b\n\t"
        "vstmia %[acc], {s0-s15}"
        : [p0] "+r"(p0), [p1] "+r"(p1), [pw] "+r"(pw), [k] "+r"(k)
        : [acc] "r"(acc)
        : "cc", "memory", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31");
    i = n & ~7;
  }
  float *a = acc, *b = acc + 8;
  for (int j = i; j < n; j++) a[0] += x0[j] * w[j];
  for (int j = i; j < n; j++) b[0] += x1[j] * w[j];
  *y0 = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
  *y1 = ((b[0] + b[4]) + (b[1] + b[5])) + ((b[2] + b[6]) + (b[3] + b[7]));
}
#endif

#ifdef JD_NEON
static float hsum(float32x4_t v) {
  float32x2_t r = vadd_f32(vget_low_f32(v), vget_high_f32(v));
  return vget_lane_f32(vpadd_f32(r, r), 0);
}

/* Two weight rows against four tokens at a time: 6 loads feed 8 multiply-adds, and the 8
 * accumulators, 2 weight and 4 activation registers fit ARMv7's 16 quad registers. */
static void matmul_neon(float *y, int ldy, const float *x, int n, int in, const jd_tensor *w,
                        float *r0, float *r1) {
  int out = (int)w->dims[0];
  for (int o = 0; o < out; o += 2) {
    int two = o + 1 < out;
    const float *w0, *w1;
    if (w->dtype == JD_F32) {
      w0 = (const float *)w->data + (size_t)o * in;
      w1 = two ? w0 + in : w0;
    } else {
      jd_dequant_row(w, o, r0);
      if (two) jd_dequant_row(w, o + 1, r1);
      w0 = r0;
      w1 = two ? r1 : r0;
    }
    int t = 0;
    for (; t + 4 <= n && in % 4 == 0; t += 4) {
      const float *x0 = x + (size_t)t * in, *x1 = x0 + in, *x2 = x1 + in, *x3 = x2 + in;
      float32x4_t a00 = vdupq_n_f32(0), a01 = a00, a02 = a00, a03 = a00;
      float32x4_t a10 = a00, a11 = a00, a12 = a00, a13 = a00;
      for (int i = 0; i < in; i += 4) {
        float32x4_t wa = vld1q_f32(w0 + i), wb = vld1q_f32(w1 + i);
        float32x4_t v0 = vld1q_f32(x0 + i), v1 = vld1q_f32(x1 + i);
        float32x4_t v2 = vld1q_f32(x2 + i), v3 = vld1q_f32(x3 + i);
        a00 = vmlaq_f32(a00, wa, v0); a01 = vmlaq_f32(a01, wa, v1);
        a02 = vmlaq_f32(a02, wa, v2); a03 = vmlaq_f32(a03, wa, v3);
        a10 = vmlaq_f32(a10, wb, v0); a11 = vmlaq_f32(a11, wb, v1);
        a12 = vmlaq_f32(a12, wb, v2); a13 = vmlaq_f32(a13, wb, v3);
      }
      float *yt = y + (size_t)t * ldy + o;
      yt[0] = hsum(a00); yt[ldy] = hsum(a01); yt[2 * ldy] = hsum(a02); yt[3 * ldy] = hsum(a03);
      if (two) {
        yt[1] = hsum(a10); yt[ldy + 1] = hsum(a11); yt[2 * ldy + 1] = hsum(a12); yt[3 * ldy + 1] = hsum(a13);
      }
    }
    for (; t < n; t++) {
      y[(size_t)t * ldy + o] = dot(x + (size_t)t * in, w0, in);
      if (two) y[(size_t)t * ldy + o + 1] = dot(x + (size_t)t * in, w1, in);
    }
  }
}
#endif

void jd_matmul(float *y, int ldy, const float *x, int n, int in, const jd_tensor *w) {
  int out = (int)w->dims[0];
#ifdef JD_NEON
  {
    float *r0 = malloc((size_t)in * 2 * sizeof(float));
    if (r0) {
      matmul_neon(y, ldy, x, n, in, w, r0, r0 + in);
      free(r0);
      return;
    }
  }
#endif
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
      int t = 0;
#ifdef JD_VFP_DOT2
      for (; t + 2 <= n; t += 2)
        dot2(x + (size_t)t * in, x + (size_t)(t + 1) * in, wr, in, &y[(size_t)t * ldy + o],
             &y[(size_t)(t + 1) * ldy + o]);
#endif
      for (; t < n; t++) y[(size_t)t * ldy + o] = dot(x + (size_t)t * in, wr, in);
    }
    free(row);
  }
}
