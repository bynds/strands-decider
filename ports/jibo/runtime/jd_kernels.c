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
#ifdef JD_NEON
  /* Sixteen or thirty-two weights per vector step instead of one. Bit for bit the scalar code
   * below: the integers convert exactly, and a scale (an fp16 value, so at least 2^-24 in
   * magnitude) times an integer from -128 to 127 is zero or a normal number, which NEON's
   * flush-to-zero cannot touch. */
  if (w->dtype == JD_Q4) {
    const unsigned char *b = (const unsigned char *)w->data + (size_t)row * nb * 18;
    const int8x16_t eight = vdupq_n_s8(8);
    for (int k = 0; k < nb; k++, b += 18) {
      float s = jd_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
      uint8x16_t q = vld1q_u8(b + 2);
      uint8x16x2_t z = vzipq_u8(vandq_u8(q, vdupq_n_u8(15)), vshrq_n_u8(q, 4)); /* low nibble first */
      float *d = dst + k * 32;
      for (int h = 0; h < 2; h++, d += 16) {
        int8x16_t v = vsubq_s8(vreinterpretq_s8_u8(z.val[h]), eight);
        int16x8_t lo = vmovl_s8(vget_low_s8(v)), hi = vmovl_s8(vget_high_s8(v));
        vst1q_f32(d, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), s));
        vst1q_f32(d + 4, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), s));
        vst1q_f32(d + 8, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), s));
        vst1q_f32(d + 12, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), s));
      }
    }
    return;
  }
  if (w->dtype == JD_Q8) {
    const unsigned char *b = (const unsigned char *)w->data + (size_t)row * nb * 34;
    for (int k = 0; k < nb; k++, b += 34) {
      float s = jd_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
      float *d = dst + k * 32;
      for (int h = 0; h < 2; h++, d += 16) {
        int8x16_t v = vld1q_s8((const int8_t *)(b + 2 + 16 * h));
        int16x8_t lo = vmovl_s8(vget_low_s8(v)), hi = vmovl_s8(vget_high_s8(v));
        vst1q_f32(d, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), s));
        vst1q_f32(d + 4, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), s));
        vst1q_f32(d + 8, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), s));
        vst1q_f32(d + 12, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), s));
      }
    }
    return;
  }
#endif
  if (w->dtype == JD_Q8) {
    const unsigned char *b = (const unsigned char *)w->data + (size_t)row * nb * 34;
    for (int k = 0; k < nb; k++, b += 34) {
      float s = jd_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
      const int8_t *q = (const int8_t *)(b + 2);
#if defined(__arm__) && defined(__ARM_FP)
      /* Per 8 weights: eight sign-extending byte loads, four paired moves into VFP registers,
       * eight conversions and eight multiplies by the scale, one store: the products of the
       * loop below, about 31 instructions where it takes 56. */
      float *d = dst + k * 32;
      int c = 4;
      unsigned r0, r1, r2, r3, r4, r5, r6, r7;
      __asm__ volatile(
          "1:\n\t"
          "ldrsb %[r0], [%[q]], #1\n\t"
          "ldrsb %[r1], [%[q]], #1\n\t"
          "ldrsb %[r2], [%[q]], #1\n\t"
          "ldrsb %[r3], [%[q]], #1\n\t"
          "ldrsb %[r4], [%[q]], #1\n\t"
          "ldrsb %[r5], [%[q]], #1\n\t"
          "ldrsb %[r6], [%[q]], #1\n\t"
          "ldrsb %[r7], [%[q]], #1\n\t"
          "vmov s0, s1, %[r0], %[r1]\n\t"
          "vmov s2, s3, %[r2], %[r3]\n\t"
          "vmov s4, s5, %[r4], %[r5]\n\t"
          "vmov s6, s7, %[r6], %[r7]\n\t"
          "vcvt.f32.s32 s0, s0\n\t"
          "vcvt.f32.s32 s1, s1\n\t"
          "vcvt.f32.s32 s2, s2\n\t"
          "vcvt.f32.s32 s3, s3\n\t"
          "vcvt.f32.s32 s4, s4\n\t"
          "vcvt.f32.s32 s5, s5\n\t"
          "vcvt.f32.s32 s6, s6\n\t"
          "vcvt.f32.s32 s7, s7\n\t"
          "vmul.f32 s0, s0, %[s]\n\t"
          "vmul.f32 s1, s1, %[s]\n\t"
          "vmul.f32 s2, s2, %[s]\n\t"
          "vmul.f32 s3, s3, %[s]\n\t"
          "vmul.f32 s4, s4, %[s]\n\t"
          "vmul.f32 s5, s5, %[s]\n\t"
          "vmul.f32 s6, s6, %[s]\n\t"
          "vmul.f32 s7, s7, %[s]\n\t"
         "vstmia %[d]!, {s0-s7}\n\t"
          "subs %[c], %[c], #1\n\t"
          "bne 1b"
          : [q] "+r"(q), [d] "+r"(d), [c] "+r"(c), [r0] "=&r"(r0), [r1] "=&r"(r1), [r2] "=&r"(r2),
            [r3] "=&r"(r3), [r4] "=&r"(r4), [r5] "=&r"(r5), [r6] "=&r"(r6), [r7] "=&r"(r7)
          : [s] "t"(s)
          : "cc", "memory", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7");
#else
      for (int i = 0; i < 32; i++) dst[k * 32 + i] = s * (float)q[i];
#endif
    }
  } else { /* JD_Q4 */
    const unsigned char *b = (const unsigned char *)w->data + (size_t)row * nb * 18;
#if defined(__arm__) && defined(__ARM_FP)
    /* Per block, the sixteen values s * (q - 8) once (the same products the loop below forms
     * one weight at a time: a load of the levels, sixteen vmul, a store), then every weight a
     * table copy: per two bytes, one halfword load, four field extracts, four indexed loads and
     * two paired stores, 13 instructions for four weights where the loop below takes 34. */
    static const float level[16] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
    float lut[16];
    float *d = dst;
    for (int k = 0; k < nb; k++, b += 18) {
      float s = jd_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
      const unsigned char *pb = b + 2;
      int c = 8;
      unsigned t, u0, u1, u2, u3;
      __asm__ volatile(
          "vldmia %[lev], {s16-s31}\n\t"
          "vmul.f32 s16, s16, %[s]\n\t"
          "vmul.f32 s17, s17, %[s]\n\t"
          "vmul.f32 s18, s18, %[s]\n\t"
          "vmul.f32 s19, s19, %[s]\n\t"
          "vmul.f32 s20, s20, %[s]\n\t"
          "vmul.f32 s21, s21, %[s]\n\t"
          "vmul.f32 s22, s22, %[s]\n\t"
          "vmul.f32 s23, s23, %[s]\n\t"
          "vmul.f32 s24, s24, %[s]\n\t"
          "vmul.f32 s25, s25, %[s]\n\t"
          "vmul.f32 s26, s26, %[s]\n\t"
          "vmul.f32 s27, s27, %[s]\n\t"
          "vmul.f32 s28, s28, %[s]\n\t"
          "vmul.f32 s29, s29, %[s]\n\t"
          "vmul.f32 s30, s30, %[s]\n\t"
          "vmul.f32 s31, s31, %[s]\n\t"
          "vstmia %[lut], {s16-s31}\n\t"
          "1:\n\t"
          "ldrh %[t], [%[pb]], #2\n\t"
          "ubfx %[u0], %[t], #0, #4\n\t"
          "ubfx %[u1], %[t], #4, #4\n\t"
          "ubfx %[u2], %[t], #8, #4\n\t"
          "lsr %[u3], %[t], #12\n\t"
          "ldr %[u0], [%[lut], %[u0], lsl #2]\n\t"
          "ldr %[u1], [%[lut], %[u1], lsl #2]\n\t"
          "ldr %[u2], [%[lut], %[u2], lsl #2]\n\t"
          "ldr %[u3], [%[lut], %[u3], lsl #2]\n\t"
          "strd %[u0], %[u1], [%[d]], #8\n\t"
          "strd %[u2], %[u3], [%[d]], #8\n\t"
          "subs %[c], %[c], #1\n\t"
          "bne 1b"
          : [pb] "+r"(pb), [d] "+r"(d), [c] "+r"(c), [t] "=&r"(t), [u0] "=&r"(u0), [u1] "=&r"(u1),
            [u2] "=&r"(u2), [u3] "=&r"(u3)
          : [lut] "r"(lut), [lev] "r"(level), [s] "t"(s)
          : "cc", "memory", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31");
    }
#else
    for (int k = 0; k < nb; k++, b += 18) {
      float s = jd_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
      for (int i = 0; i < 16; i++) {
        dst[k * 32 + 2 * i] = s * (float)((int)(b[2 + i] & 15) - 8);
        dst[k * 32 + 2 * i + 1] = s * (float)((int)(b[2 + i] >> 4) - 8);
      }
    }
#endif
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
    /* Sixteen inputs per iteration (the 8-input body twice), then a last 8 if n / 8 is odd:
     * the loop's subs and bne once per 32 multiply-adds instead of per 16. */
    int k16 = k / 2, odd = k & 1;
    __asm__ volatile(
        "vldmia %[acc], {s0-s15}\n\t"
        "cmp %[k16], #0\n\t"
        "beq 2f\n\t"
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
        "subs %[k16], %[k16], #1\n\t"
        "bne 1b\n\t"
        "2:\n\t"
        "cmp %[odd], #0\n\t"
        "beq 3f\n\t"
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
        "3:\n\t"
        "vstmia %[acc], {s0-s15}"
        : [p0] "+r"(p0), [p1] "+r"(p1), [pw] "+r"(pw), [k16] "+r"(k16)
        : [acc] "r"(acc), [odd] "r"(odd)
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
    if (in % 8 == 0)
      for (; t + 4 <= n; t += 4) {
        /* The tile below in assembly: per 8 inputs, one 4-register load per weight row and
         * per token (post-increment) and sixteen non-fused vmla, 24 instructions where GCC's
         * intrinsics take 54. Each accumulator lane adds inputs i, then i + 4: the order of the
         * loop below, so the sums are the same to the bit. Accumulators: row 0 in q0-q3,
         * row 1 in q12-q15; weights q8-q11, a token's inputs q4-q5 (d8-d11, callee-saved, so
         * clobbers that GCC saves once). */
        const float *x0 = x + (size_t)t * in, *x1 = x0 + in, *x2 = x1 + in, *x3 = x2 + in;
        const float *pw0 = w0, *pw1 = w1;
        float acc[32], *pa = acc;
        int k = in / 8;
        __asm__ volatile(
            "vmov.i32 q0, #0\n\t"
            "vmov.i32 q1, #0\n\t"
            "vmov.i32 q2, #0\n\t"
            "vmov.i32 q3, #0\n\t"
            "vmov.i32 q12, #0\n\t"
            "vmov.i32 q13, #0\n\t"
            "vmov.i32 q14, #0\n\t"
            "vmov.i32 q15, #0\n\t"
            "1:\n\t"
            "vld1.32 {d16-d19}, [%[w0]]!\n\t"
            "vld1.32 {d20-d23}, [%[w1]]!\n\t"
            "vld1.32 {d8-d11}, [%[x0]]!\n\t"
            "vmla.f32 q0, q8, q4\n\t"
            "vmla.f32 q12, q10, q4\n\t"
            "vmla.f32 q0, q9, q5\n\t"
            "vmla.f32 q12, q11, q5\n\t"
            "vld1.32 {d8-d11}, [%[x1]]!\n\t"
            "vmla.f32 q1, q8, q4\n\t"
            "vmla.f32 q13, q10, q4\n\t"
            "vmla.f32 q1, q9, q5\n\t"
            "vmla.f32 q13, q11, q5\n\t"
            "vld1.32 {d8-d11}, [%[x2]]!\n\t"
            "vmla.f32 q2, q8, q4\n\t"
            "vmla.f32 q14, q10, q4\n\t"
            "vmla.f32 q2, q9, q5\n\t"
            "vmla.f32 q14, q11, q5\n\t"
            "vld1.32 {d8-d11}, [%[x3]]!\n\t"
            "vmla.f32 q3, q8, q4\n\t"
            "vmla.f32 q15, q10, q4\n\t"
            "vmla.f32 q3, q9, q5\n\t"
            "vmla.f32 q15, q11, q5\n\t"
            "subs %[k], %[k], #1\n\t"
            "bne 1b\n\t"
            "vst1.32 {d0-d3}, [%[acc]]!\n\t"
            "vst1.32 {d4-d7}, [%[acc]]!\n\t"
            "vst1.32 {d24-d27}, [%[acc]]!\n\t"
            "vst1.32 {d28-d31}, [%[acc]]"
            : [w0] "+r"(pw0), [w1] "+r"(pw1), [x0] "+r"(x0), [x1] "+r"(x1), [x2] "+r"(x2), [x3] "+r"(x3),
              [k] "+r"(k), [acc] "+r"(pa), "=m"(acc)
            :
            : "cc", "memory", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
              "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
              "d29", "d30", "d31");
        float *yt = y + (size_t)t * ldy + o;
        yt[0] = hsum(vld1q_f32(acc)); yt[ldy] = hsum(vld1q_f32(acc + 4));
        yt[2 * ldy] = hsum(vld1q_f32(acc + 8)); yt[3 * ldy] = hsum(vld1q_f32(acc + 12));
        if (two) {
          yt[1] = hsum(vld1q_f32(acc + 16)); yt[ldy + 1] = hsum(vld1q_f32(acc + 20));
          yt[2 * ldy + 1] = hsum(vld1q_f32(acc + 24)); yt[3 * ldy + 1] = hsum(vld1q_f32(acc + 28));
        }
      }
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
