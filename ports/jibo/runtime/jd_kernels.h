/* Matrix kernels of the CPU backend. */
#ifndef JD_KERNELS_H
#define JD_KERNELS_H

#include "jd_model.h"

/* y[t * ldy + o] = sum_i x[t * in + i] * W[o, i] for t < n and every row o of W
 * (W is out x in, stored f32, q8 or q4). */
void jd_matmul(float *y, int ldy, const float *x, int n, int in, const jd_tensor *w);

/* Dequantise one row of W into dst (in floats). */
void jd_dequant_row(const jd_tensor *w, int row, float *dst);

float jd_f16_to_f32(uint16_t h);

#endif
