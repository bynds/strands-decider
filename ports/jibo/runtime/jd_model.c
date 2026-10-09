#include "jd_model.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "jd_kernels.h"
#include "jd_prof.h"

/* ---- loading ----------------------------------------------------------------------------- */

static uint32_t rd32(const unsigned char *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const unsigned char *p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }

const char *jd_model_meta(const jd_model *m, const char *key) {
  size_t kl = strlen(key);
  const char *p = m->meta;
  while (p && *p) {
    const char *nl = strchr(p, '\n');
    if (strncmp(p, key, kl) == 0 && p[kl] == '=') return p + kl + 1;
    p = nl ? nl + 1 : NULL;
  }
  return NULL;
}

static int meta_int(const jd_model *m, const char *k, int *out) {
  const char *v = jd_model_meta(m, k);
  if (!v) return -1;
  *out = (int)strtol(v, NULL, 10);
  return 0;
}
static int meta_float(const jd_model *m, const char *k, float *out) {
  const char *v = jd_model_meta(m, k);
  if (!v) return -1;
  *out = strtof(v, NULL);
  return 0;
}

typedef struct {
  const char *name;
  jd_tensor *dst;
  int required;
} want;

static int find_tensor(const unsigned char *base, size_t len, uint32_t n, size_t table_off,
                       const char *name, jd_tensor *out) {
  size_t off = table_off, nl = strlen(name);
  for (uint32_t i = 0; i < n; i++) {
    if (off + 2 > len) return -1;
    uint32_t L = (uint32_t)base[off] | ((uint32_t)base[off + 1] << 8);
    const char *nm = (const char *)base + off + 2;
    off += 2 + L;
    int dtype = base[off], ndim = base[off + 1];
    off += 2;
    uint32_t dims[4] = {1, 1, 1, 1};
    for (int d = 0; d < ndim; d++) dims[d] = rd32(base + off + 4 * d);
    off += 4 * (size_t)ndim;
    uint64_t data = rd64(base + off), nbytes = rd64(base + off + 8);
    off += 16;
    if (L == nl && memcmp(nm, name, nl) == 0) {
      if (data + nbytes > len || ndim > 4) return -1;
      out->dtype = dtype;
      out->ndim = ndim;
      memcpy(out->dims, dims, sizeof(dims));
      out->data = base + data;
      out->nbytes = nbytes;
      return 0;
    }
  }
  return -1;
}

int jd_model_load(jd_model *m, const char *path) {
  memset(m, 0, sizeof(*m));
  int fd = open(path, O_RDONLY);
  if (fd < 0) return -1;
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < 64) { close(fd); return -1; }
  void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) return -1;
  m->map = map;
  m->map_len = (size_t)st.st_size;
  const unsigned char *p = map;
  if (memcmp(p, "JDW1", 4) != 0 || rd32(p + 4) != 1) goto fail;
  uint32_t meta_len = rd32(p + 8), n_tensors = rd32(p + 12);
  m->meta = malloc(meta_len + 1);
  if (!m->meta) goto fail;
  memcpy(m->meta, p + 24, meta_len);
  m->meta[meta_len] = 0;
  size_t table = 24 + meta_len;
  m->n_tensors = n_tensors;
  m->table_off = table;

  jd_config *c = &m->c;
  if (meta_int(m, "vocab_size", &c->vocab) || meta_int(m, "hidden_size", &c->hidden) ||
      meta_int(m, "intermediate_size", &c->inter) || meta_int(m, "num_layers", &c->n_layers) ||
      meta_int(m, "num_heads", &c->n_heads) || meta_int(m, "num_kv_heads", &c->n_kv) ||
      meta_int(m, "head_dim", &c->head_dim) || meta_int(m, "rotary_dim", &c->rot_dim) ||
      meta_float(m, "rope_theta", &c->rope_theta) || meta_float(m, "rms_eps", &c->eps) ||
      meta_int(m, "lin_num_k_heads", &c->lin_nk) || meta_int(m, "lin_num_v_heads", &c->lin_nv) ||
      meta_int(m, "lin_k_dim", &c->lin_dk) || meta_int(m, "lin_v_dim", &c->lin_dv) ||
      meta_int(m, "conv_kernel", &c->conv_k) || meta_int(m, "max_length", &c->max_length) ||
      meta_int(m, "pointer_dim", &c->pointer_dim) || meta_float(m, "head_ln_eps", &c->head_ln_eps) ||
      meta_float(m, "temperature", &c->temperature) ||
      meta_float(m, "temperature_noul", &c->temp_noul) ||
      meta_float(m, "temperature_choice", &c->temp_choice) ||
      meta_float(m, "temperature_score", &c->temp_score) ||
      meta_float(m, "ordinal_smoothing", &c->ordinal_smoothing))
    goto fail;
  if (c->lin_nv % c->lin_nk || c->n_heads % c->n_kv || c->rot_dim % 2 || c->conv_k < 2 || c->conv_k > 8) goto fail;
  const char *types = jd_model_meta(m, "layer_types");
  if (!types) goto fail;
  m->layers = calloc((size_t)c->n_layers, sizeof(jd_layer));
  if (!m->layers) goto fail;

#define T(name, dst) \
  if (find_tensor(p, m->map_len, n_tensors, table, name, dst) != 0) goto fail
  T("embed", &m->embed);
  T("final_norm", &m->final_norm);
  T("head.ln_w", &m->head_ln_w);
  T("head.ln_b", &m->head_ln_b);
  T("head.q_w", &m->head_q_w);
  T("head.q_b", &m->head_q_b);
  T("head.k_w", &m->head_k_w);
  T("head.k_b", &m->head_k_b);
  char name[64];
  for (int i = 0; i < c->n_layers; i++) {
    jd_layer *L = &m->layers[i];
    char t = types[2 * i];
    if (t != 'L' && t != 'A') goto fail;
    L->type = t == 'L' ? JD_LAYER_LINEAR : JD_LAYER_ATTN;
#define LT(field, suffix) \
  snprintf(name, sizeof(name), "l%d." suffix, i); \
  T(name, &L->field)
    LT(in_norm, "in_norm"); LT(post_norm, "post_norm");
    LT(gate, "gate"); LT(up, "up"); LT(down, "down");
    if (L->type == JD_LAYER_LINEAR) {
      LT(qkv, "qkv"); LT(z, "z"); LT(b, "b"); LT(a, "a"); LT(conv, "conv");
      LT(dt_bias, "dt_bias"); LT(a_log, "a_log"); LT(gnorm, "gnorm"); LT(out, "out");
    } else {
      LT(q, "q"); LT(k, "k"); LT(v, "v"); LT(o, "o"); LT(q_norm, "q_norm"); LT(k_norm, "k_norm");
    }
#undef LT
  }
#undef T
  /* Rotary inverse frequencies, as transformers computes them in fp32. */
  int nf = c->rot_dim / 2;
  m->rope_inv_freq = malloc((size_t)nf * sizeof(float));
  if (!m->rope_inv_freq) goto fail;
  for (int i = 0; i < nf; i++)
    m->rope_inv_freq[i] = 1.0f / powf(c->rope_theta, (float)(2 * i) / (float)c->rot_dim);
  return 0;
fail:
  jd_model_free(m);
  return -1;
}

int jd_model_find(const jd_model *m, const char *name, jd_tensor *out) {
  return find_tensor(m->map, m->map_len, m->n_tensors, m->table_off, name, out);
}

void jd_model_free(jd_model *m) {
  if (m->map) munmap(m->map, m->map_len);
  free(m->layers);
  free(m->meta);
  free(m->rope_inv_freq);
  memset(m, 0, sizeof(*m));
}

/* ---- state ------------------------------------------------------------------------------- */

static int conv_dim(const jd_config *c) { return 2 * c->lin_nk * c->lin_dk + c->lin_nv * c->lin_dv; }

int jd_state_init(jd_state *s, const jd_model *m, int cap) {
  const jd_config *c = &m->c;
  memset(s, 0, sizeof(*s));
  s->cap = cap;
  s->conv = calloc((size_t)c->n_layers, sizeof(float *));
  s->rec = calloc((size_t)c->n_layers, sizeof(float *));
  s->kc = calloc((size_t)c->n_layers, sizeof(float *));
  s->vc = calloc((size_t)c->n_layers, sizeof(float *));
  if (!s->conv || !s->rec || !s->kc || !s->vc) return -1;
  for (int i = 0; i < c->n_layers; i++) {
    if (m->layers[i].type == JD_LAYER_LINEAR) {
      s->conv[i] = calloc((size_t)(c->conv_k - 1) * (size_t)conv_dim(c), sizeof(float));
      s->rec[i] = calloc((size_t)c->lin_nv * c->lin_dk * c->lin_dv, sizeof(float));
      if (!s->conv[i] || !s->rec[i]) return -1;
    } else {
      size_t n = (size_t)cap * c->n_kv * c->head_dim;
      s->kc[i] = malloc(n * sizeof(float));
      s->vc[i] = malloc(n * sizeof(float));
      if (!s->kc[i] || !s->vc[i]) return -1;
    }
  }
  return 0;
}

void jd_state_free(jd_state *s, const jd_model *m) {
  for (int i = 0; s->conv && i < m->c.n_layers; i++) {
    free(s->conv[i]); free(s->rec[i]); free(s->kc[i]); free(s->vc[i]);
  }
  free(s->conv); free(s->rec); free(s->kc); free(s->vc);
  memset(s, 0, sizeof(*s));
}

void jd_state_reset(jd_state *s, const jd_model *m) {
  const jd_config *c = &m->c;
  JD_PB(JD_P_STATE);
  s->pos = 0;
  for (int i = 0; i < c->n_layers; i++) {
    if (s->conv[i]) memset(s->conv[i], 0, (size_t)(c->conv_k - 1) * conv_dim(c) * sizeof(float));
    if (s->rec[i]) memset(s->rec[i], 0, (size_t)c->lin_nv * c->lin_dk * c->lin_dv * sizeof(float));
  }
  JD_PE();
}

void jd_state_copy(jd_state *dst, const jd_state *src, const jd_model *m) {
  const jd_config *c = &m->c;
  JD_PB(JD_P_STATE);
  dst->pos = src->pos;
  for (int i = 0; i < c->n_layers; i++) {
    if (src->conv[i]) {
      memcpy(dst->conv[i], src->conv[i], (size_t)(c->conv_k - 1) * conv_dim(c) * sizeof(float));
      memcpy(dst->rec[i], src->rec[i], (size_t)c->lin_nv * c->lin_dk * c->lin_dv * sizeof(float));
    } else {
      size_t n = (size_t)src->pos * c->n_kv * c->head_dim * sizeof(float);
      memcpy(dst->kc[i], src->kc[i], n);
      memcpy(dst->vc[i], src->vc[i], n);
    }
  }
  JD_PE();
}

/* ---- the forward ------------------------------------------------------------------------- */

static float silu(float x) { return x / (1.0f + expf(-x)); }
static float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
static float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); } /* torch's threshold 20 */

/* y = rmsnorm(x) * w over rows of width d. */
static void rmsnorm(float *y, const float *x, const float *w, int rows, int d, float eps) {
  for (int r = 0; r < rows; r++) {
    const float *xr = x + (size_t)r * d;
    float *yr = y + (size_t)r * d;
    float ss = 0.0f;
    for (int i = 0; i < d; i++) ss += xr[i] * xr[i];
    float inv = 1.0f / sqrtf(ss / (float)d + eps);
    for (int i = 0; i < d; i++) yr[i] = xr[i] * inv * w[i];
  }
}

static void embed_rows(const jd_model *m, const int32_t *ids, int n, float *x) {
  int H = m->c.hidden;
  for (int t = 0; t < n; t++) {
    float *xr = x + (size_t)t * H;
    if (m->embed.dtype == JD_BF16) {
      const uint16_t *row = (const uint16_t *)m->embed.data + (size_t)ids[t] * H;
      for (int i = 0; i < H; i++) {
        uint32_t b = (uint32_t)row[i] << 16;
        memcpy(&xr[i], &b, 4);
      }
    } else {
      memcpy(xr, (const float *)m->embed.data + (size_t)ids[t] * H, (size_t)H * sizeof(float));
    }
  }
}

typedef struct {
  float *h, *mix, *big1, *big2, *small;
} scratch;

#if defined(__arm__) && defined(__ARM_FP)
/* The two sweeps of the delta rule over eight state columns (rows of `stride` floats, `dk` of
 * them), in VFP assembly for both ARM builds: per row, one 8-register load and store, eight vmul
 * or vmla for the state and eight vmla for the sums. GCC's code takes 36 instructions a row,
 * and with NEON enabled 43, bouncing the stores through the stack. The arithmetic, its operand
 * order and the order of each column's sum are the C code's.
 *
 * rec_decay: S *= decay; k += S * kn[i]    (k[8] in and out)
 * rec_update: S += kn[i] * c; o += S * qn[i] (c[8] in, o[8] in and out) */
static void rec_decay(float *S, int stride, const float *kn, int dk, const float *decay, float *k) {
  __asm__ volatile(
      "vldmia %[k], {s0-s7}\n\t"
      "vldr s16, [%[decay]]\n\t"
      "1:\n\t"
      "vldmia %[S], {s8-s15}\n\t"
      "vldmia %[kn]!, {s17}\n\t"
      "vmul.f32 s8, s8, s16\n\t"
      "vmul.f32 s9, s9, s16\n\t"
      "vmul.f32 s10, s10, s16\n\t"
      "vmul.f32 s11, s11, s16\n\t"
      "vmul.f32 s12, s12, s16\n\t"
      "vmul.f32 s13, s13, s16\n\t"
      "vmul.f32 s14, s14, s16\n\t"
      "vmul.f32 s15, s15, s16\n\t"
      "vstmia %[S], {s8-s15}\n\t"
      "vmla.f32 s0, s8, s17\n\t"
      "vmla.f32 s1, s9, s17\n\t"
      "vmla.f32 s2, s10, s17\n\t"
      "vmla.f32 s3, s11, s17\n\t"
      "vmla.f32 s4, s12, s17\n\t"
      "vmla.f32 s5, s13, s17\n\t"
      "vmla.f32 s6, s14, s17\n\t"
      "vmla.f32 s7, s15, s17\n\t"
      "add %[S], %[S], %[stride]\n\t"
      "subs %[n], %[n], #1\n\t"
      "bne 1b\n\t"
      "vstmia %[k], {s0-s7}"
      : [S] "+r"(S), [kn] "+r"(kn), [n] "+r"(dk)
      : [k] "r"(k), [decay] "r"(decay), [stride] "r"(stride * 4)
      : "cc", "memory", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31");
}

static void rec_update(float *S, int stride, const float *kn, const float *qn, int dk, const float *c, float *o) {
  __asm__ volatile(
      "vldmia %[o], {s0-s7}\n\t"
      "vldmia %[c], {s18-s25}\n\t"
      "1:\n\t"
      "vldmia %[S], {s8-s15}\n\t"
      "vldmia %[kn]!, {s16}\n\t"
      "vldmia %[qn]!, {s17}\n\t"
      "vmla.f32 s8, s16, s18\n\t"
      "vmla.f32 s9, s16, s19\n\t"
      "vmla.f32 s10, s16, s20\n\t"
      "vmla.f32 s11, s16, s21\n\t"
      "vmla.f32 s12, s16, s22\n\t"
      "vmla.f32 s13, s16, s23\n\t"
      "vmla.f32 s14, s16, s24\n\t"
      "vmla.f32 s15, s16, s25\n\t"
      "vstmia %[S], {s8-s15}\n\t"
      "vmla.f32 s0, s8, s17\n\t"
      "vmla.f32 s1, s9, s17\n\t"
      "vmla.f32 s2, s10, s17\n\t"
      "vmla.f32 s3, s11, s17\n\t"
      "vmla.f32 s4, s12, s17\n\t"
      "vmla.f32 s5, s13, s17\n\t"
      "vmla.f32 s6, s14, s17\n\t"
      "vmla.f32 s7, s15, s17\n\t"
      "add %[S], %[S], %[stride]\n\t"
      "subs %[n], %[n], #1\n\t"
      "bne 1b\n\t"
      "vstmia %[o], {s0-s7}"
      : [S] "+r"(S), [kn] "+r"(kn), [qn] "+r"(qn), [n] "+r"(dk)
      : [o] "r"(o), [c] "r"(c), [stride] "r"(stride * 4)
      : "cc", "memory", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31");
}
#endif

static void gated_deltanet(const jd_model *m, const jd_layer *L, int li, jd_state *s, const float *h,
                           int n, float *out, scratch *sc) {
  const jd_config *c = &m->c;
  int H = c->hidden, kd = c->lin_nk * c->lin_dk, vd = c->lin_nv * c->lin_dv, cd = conv_dim(c);
  int K = c->conv_k, nv = c->lin_nv, dk = c->lin_dk, dv = c->lin_dv, rep = c->lin_nv / c->lin_nk;
  float *mixed = sc->big1;   /* n x cd */
  float *z = sc->big2;       /* n x vd */
  float *ab = sc->small;     /* n x 2nv: b then a */
  JD_PB(JD_P_LIN_PROJ);
  jd_matmul(mixed, cd, h, n, H, &L->qkv);
  jd_matmul(z, vd, h, n, H, &L->z);
  jd_matmul(ab, 2 * nv, h, n, H, &L->b);
  jd_matmul(ab + nv, 2 * nv, h, n, H, &L->a);
  JD_PE();

  /* causal depthwise convolution over [state; mixed], then SiLU; the state keeps the last K-1 inputs */
  const float *w = (const float *)L->conv.data; /* cd x K */
  float *cs = s->conv[li];                      /* (K-1) x cd */
  float *conv_out = sc->mix;                    /* n x cd (mix is n x max(H, cd)) */
  JD_PB(JD_P_LIN_CONV);
  for (int t = 0; t < n; t++) {
    /* the K input rows of this step, resolved once instead of per channel and tap */
    const float *r[8];
    for (int j = 0; j < K; j++) {
      int src = t - (K - 1) + j; /* index into mixed; negative reads the state */
      r[j] = src >= 0 ? mixed + (size_t)src * cd : cs + (size_t)(K - 1 + src) * cd;
    }
    float *co = conv_out + (size_t)t * cd;
    if (K == 4) {
      const float *r0 = r[0], *r1 = r[1], *r2 = r[2], *r3 = r[3];
      for (int ch = 0; ch < cd; ch++) {
        const float *wc = w + (size_t)ch * 4;
        float acc = 0.0f;
        acc += wc[0] * r0[ch];
        acc += wc[1] * r1[ch];
        acc += wc[2] * r2[ch];
        acc += wc[3] * r3[ch];
        co[ch] = silu(acc);
      }
    } else {
      for (int ch = 0; ch < cd; ch++) {
        float acc = 0.0f;
        for (int j = 0; j < K; j++) acc += w[(size_t)ch * K + j] * r[j][ch];
        co[ch] = silu(acc);
      }
    }
  }
  /* new conv state: the last K-1 inputs of [state; mixed] */
  for (int j = 0; j < K - 1; j++) {
    int src = n - (K - 1) + j;
    float *dst = cs + (size_t)j * cd;
    if (src >= 0) memcpy(dst, mixed + (size_t)src * cd, (size_t)cd * sizeof(float));
    else memmove(dst, cs + (size_t)(K - 1 + src) * cd, (size_t)cd * sizeof(float));
  }
  /* (when n < K-1 the memmove reads entry j+n, which this loop has not overwritten yet) */
  JD_PE();

  const float *dt_bias = (const float *)L->dt_bias.data, *a_log = (const float *)L->a_log.data;
  const float *gw = (const float *)L->gnorm.data;
  float *S = s->rec[li];
  float qn[512], kn[512], kv_mem[512], o[512];
  float qscale = 1.0f / sqrtf((float)dk);
  JD_PB(JD_P_LIN_REC);
  for (int t = 0; t < n; t++) {
    const float *row = conv_out + (size_t)t * cd;
    for (int hv = 0; hv < nv; hv++) {
      int hk = hv / rep;
      const float *q = row + (size_t)hk * dk, *k = row + kd + (size_t)hk * dk;
      const float *v = row + 2 * kd + (size_t)hv * dv;
      float sq = 0, sk = 0;
      for (int i = 0; i < dk; i++) { sq += q[i] * q[i]; sk += k[i] * k[i]; }
      float iq = 1.0f / sqrtf(sq + 1e-6f), ik = 1.0f / sqrtf(sk + 1e-6f);
      for (int i = 0; i < dk; i++) { qn[i] = q[i] * iq * qscale; kn[i] = k[i] * ik; }
      float beta = sigmoidf_(ab[(size_t)t * 2 * nv + hv]);
      float g = -expf(a_log[hv]) * softplusf(ab[(size_t)t * 2 * nv + nv + hv] + dt_bias[hv]);
      float decay = expf(g);
      float *Sh = S + (size_t)hv * dk * dv;
      if (dv % 8 == 0) {
        /* The two sweeps over the state below, eight columns at a time, with the columns'
         * running sums in named registers instead of memory: each kv_mem[j] and o[j] still adds
         * its terms over i in the same order. */
        for (int jb = 0; jb < dv; jb += 8) {
#if defined(__arm__) && defined(__ARM_FP)
          float kb[8] = {0, 0, 0, 0, 0, 0, 0, 0};
          rec_decay(Sh + jb, dv, kn, dk, &decay, kb);
          memcpy(kv_mem + jb, kb, sizeof kb);
#else
          float k0 = 0, k1 = 0, k2 = 0, k3 = 0, k4 = 0, k5 = 0, k6 = 0, k7 = 0;
          for (int i = 0; i < dk; i++) {
            float *Si = Sh + (size_t)i * dv + jb, kni = kn[i];
            float s0 = Si[0] * decay, s1 = Si[1] * decay, s2 = Si[2] * decay, s3 = Si[3] * decay;
            float s4 = Si[4] * decay, s5 = Si[5] * decay, s6 = Si[6] * decay, s7 = Si[7] * decay;
            Si[0] = s0; Si[1] = s1; Si[2] = s2; Si[3] = s3; Si[4] = s4; Si[5] = s5; Si[6] = s6; Si[7] = s7;
            k0 += s0 * kni; k1 += s1 * kni; k2 += s2 * kni; k3 += s3 * kni;
            k4 += s4 * kni; k5 += s5 * kni; k6 += s6 * kni; k7 += s7 * kni;
          }
          kv_mem[jb] = k0; kv_mem[jb + 1] = k1; kv_mem[jb + 2] = k2; kv_mem[jb + 3] = k3;
          kv_mem[jb + 4] = k4; kv_mem[jb + 5] = k5; kv_mem[jb + 6] = k6; kv_mem[jb + 7] = k7;
#endif
        }
        for (int j = 0; j < dv; j++) kv_mem[j] = (v[j] - kv_mem[j]) * beta; /* delta */
        for (int jb = 0; jb < dv; jb += 8) {
#if defined(__arm__) && defined(__ARM_FP)
          float ob[8] = {0, 0, 0, 0, 0, 0, 0, 0};
          rec_update(Sh + jb, dv, kn, qn, dk, kv_mem + jb, ob);
          memcpy(o + jb, ob, sizeof ob);
#else
          float c0 = kv_mem[jb], c1 = kv_mem[jb + 1], c2 = kv_mem[jb + 2], c3 = kv_mem[jb + 3];
          float c4 = kv_mem[jb + 4], c5 = kv_mem[jb + 5], c6 = kv_mem[jb + 6], c7 = kv_mem[jb + 7];
          float o0 = 0, o1 = 0, o2 = 0, o3 = 0, o4 = 0, o5 = 0, o6 = 0, o7 = 0;
          for (int i = 0; i < dk; i++) {
            float *Si = Sh + (size_t)i * dv + jb, kni = kn[i], qni = qn[i];
            float s0 = Si[0] + kni * c0, s1 = Si[1] + kni * c1, s2 = Si[2] + kni * c2, s3 = Si[3] + kni * c3;
            float s4 = Si[4] + kni * c4, s5 = Si[5] + kni * c5, s6 = Si[6] + kni * c6, s7 = Si[7] + kni * c7;
            Si[0] = s0; Si[1] = s1; Si[2] = s2; Si[3] = s3; Si[4] = s4; Si[5] = s5; Si[6] = s6; Si[7] = s7;
            o0 += s0 * qni; o1 += s1 * qni; o2 += s2 * qni; o3 += s3 * qni;
            o4 += s4 * qni; o5 += s5 * qni; o6 += s6 * qni; o7 += s7 * qni;
          }
          o[jb] = o0; o[jb + 1] = o1; o[jb + 2] = o2; o[jb + 3] = o3;
          o[jb + 4] = o4; o[jb + 5] = o5; o[jb + 6] = o6; o[jb + 7] = o7;
#endif
        }
      } else {
        for (int j = 0; j < dv; j++) kv_mem[j] = 0.0f;
        for (int i = 0; i < dk; i++) {
          float *Si = Sh + (size_t)i * dv;
          for (int j = 0; j < dv; j++) {
            Si[j] *= decay;
            kv_mem[j] += Si[j] * kn[i];
          }
        }
        for (int j = 0; j < dv; j++) kv_mem[j] = (v[j] - kv_mem[j]) * beta; /* delta */
        for (int j = 0; j < dv; j++) o[j] = 0.0f;
        for (int i = 0; i < dk; i++) {
          float *Si = Sh + (size_t)i * dv;
          for (int j = 0; j < dv; j++) {
            Si[j] += kn[i] * kv_mem[j];
            o[j] += Si[j] * qn[i];
          }
        }
      }
      /* gated RMSNorm over the head, then into the output row */
      float ss = 0;
      for (int j = 0; j < dv; j++) ss += o[j] * o[j];
      float inv = 1.0f / sqrtf(ss / (float)dv + c->eps);
      const float *zh = z + (size_t)t * vd + (size_t)hv * dv;
      for (int j = 0; j < dv; j++) o[j] = gw[j] * (o[j] * inv) * silu(zh[j]);
      memcpy(z + (size_t)t * vd + (size_t)hv * dv, o, (size_t)dv * sizeof(float));
    }
  }
  JD_PE();
  JD_PB(JD_P_LIN_OUT);
  jd_matmul(out, H, z, n, vd, &L->out);
  JD_PE();
}

static void rope(const jd_model *m, float *x, int pos) {
  int half = m->c.rot_dim / 2;
  for (int i = 0; i < half; i++) {
    float f = (float)pos * m->rope_inv_freq[i];
    float cs = cosf(f), sn = sinf(f);
    float a = x[i], b = x[i + half];
    x[i] = a * cs - b * sn;
    x[i + half] = b * cs + a * sn;
  }
}

#if defined(__arm__) && defined(__ARM_FP)
/* Attention in VFP assembly. attn_qk2: two scores, q.k0 and q.k1 over n (a multiple of 8)
 * inputs, the query loaded once for both; each score is one accumulator adding q[i] k[i] in
 * order of i, as the C loop does. attn_pv8: eight output columns o[0..8) += w[j] v_j[0..8) for
 * j < np, where v_j = v + j * stride, positions two at a time; each column adds its terms in
 * order of j. */
static void attn_qk2(const float *q, const float *k0, const float *k1, int n, float *d) {
  int c = n / 8;
  __asm__ volatile(
      "vldmia %[d], {s0-s1}\n\t"
      "1:\n\t"
      "vldmia %[q]!, {s16-s23}\n\t"
      "vldmia %[k0]!, {s24-s31}\n\t"
      "vmla.f32 s0, s16, s24\n\t"
      "vmla.f32 s0, s17, s25\n\t"
      "vmla.f32 s0, s18, s26\n\t"
      "vmla.f32 s0, s19, s27\n\t"
      "vmla.f32 s0, s20, s28\n\t"
      "vmla.f32 s0, s21, s29\n\t"
      "vmla.f32 s0, s22, s30\n\t"
      "vmla.f32 s0, s23, s31\n\t"
      "vldmia %[k1]!, {s24-s31}\n\t"
      "vmla.f32 s1, s16, s24\n\t"
      "vmla.f32 s1, s17, s25\n\t"
      "vmla.f32 s1, s18, s26\n\t"
      "vmla.f32 s1, s19, s27\n\t"
      "vmla.f32 s1, s20, s28\n\t"
      "vmla.f32 s1, s21, s29\n\t"
      "vmla.f32 s1, s22, s30\n\t"
      "vmla.f32 s1, s23, s31\n\t"
      "subs %[c], %[c], #1\n\t"
      "bne 1b\n\t"
      "vstmia %[d], {s0-s1}"
      : [q] "+r"(q), [k0] "+r"(k0), [k1] "+r"(k1), [c] "+r"(c)
      : [d] "r"(d)
      : "cc", "memory", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31");
}

static void attn_pv8(const float *v, int stride, const float *w, int np, float *o) {
  int pairs = np / 2, odd = np & 1;
  __asm__ volatile(
      "vldmia %[o], {s0-s7}\n\t"
      "cmp %[pairs], #0\n\t"
      "beq 2f\n\t"
      "1:\n\t"
      "vldmia %[w]!, {s8-s9}\n\t"
      "vldmia %[v], {s16-s23}\n\t"
      "add %[v], %[v], %[stride]\n\t"
      "vldmia %[v], {s24-s31}\n\t"
      "add %[v], %[v], %[stride]\n\t"
      "vmla.f32 s0, s8, s16\n\t"
      "vmla.f32 s1, s8, s17\n\t"
      "vmla.f32 s2, s8, s18\n\t"
      "vmla.f32 s3, s8, s19\n\t"
      "vmla.f32 s4, s8, s20\n\t"
      "vmla.f32 s5, s8, s21\n\t"
      "vmla.f32 s6, s8, s22\n\t"
      "vmla.f32 s7, s8, s23\n\t"
      "vmla.f32 s0, s9, s24\n\t"
      "vmla.f32 s1, s9, s25\n\t"
      "vmla.f32 s2, s9, s26\n\t"
      "vmla.f32 s3, s9, s27\n\t"
      "vmla.f32 s4, s9, s28\n\t"
      "vmla.f32 s5, s9, s29\n\t"
      "vmla.f32 s6, s9, s30\n\t"
      "vmla.f32 s7, s9, s31\n\t"
      "subs %[pairs], %[pairs], #1\n\t"
      "bne 1b\n\t"
      "2:\n\t"
      "cmp %[odd], #0\n\t"
      "beq 3f\n\t"
      "vldmia %[w]!, {s8}\n\t"
      "vldmia %[v], {s16-s23}\n\t"
      "vmla.f32 s0, s8, s16\n\t"
      "vmla.f32 s1, s8, s17\n\t"
      "vmla.f32 s2, s8, s18\n\t"
      "vmla.f32 s3, s8, s19\n\t"
      "vmla.f32 s4, s8, s20\n\t"
      "vmla.f32 s5, s8, s21\n\t"
      "vmla.f32 s6, s8, s22\n\t"
      "vmla.f32 s7, s8, s23\n\t"
      "3:\n\t"
      "vstmia %[o], {s0-s7}"
      : [v] "+r"(v), [w] "+r"(w), [pairs] "+r"(pairs)
      : [o] "r"(o), [odd] "r"(odd), [stride] "r"(stride * 4)
      : "cc", "memory", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31");
}
#endif

static void attention(const jd_model *m, const jd_layer *L, int li, jd_state *s, const float *h,
                      int n, float *out, scratch *sc) {
  const jd_config *c = &m->c;
  int H = c->hidden, nh = c->n_heads, nkv = c->n_kv, hd = c->head_dim, rep = nh / nkv;
  float *qg = sc->big1;  /* n x (2 nh hd): per head, query then gate */
  float *kv = sc->big2;  /* n x (2 nkv hd): keys then values */
  JD_PB(JD_P_ATTN_PROJ);
  jd_matmul(qg, 2 * nh * hd, h, n, H, &L->q);
  jd_matmul(kv, 2 * nkv * hd, h, n, H, &L->k);
  jd_matmul(kv + nkv * hd, 2 * nkv * hd, h, n, H, &L->v);
  JD_PE();
  const float *qnw = (const float *)L->q_norm.data, *knw = (const float *)L->k_norm.data;
  float *kc = s->kc[li], *vc = s->vc[li];
  JD_PB(JD_P_ATTN_ROPE);
  for (int t = 0; t < n; t++) {
    int pos = s->pos + t;
    for (int hh = 0; hh < nh; hh++) {
      float *q = qg + (size_t)t * 2 * nh * hd + (size_t)hh * 2 * hd;
      rmsnorm(q, q, qnw, 1, hd, c->eps);
      rope(m, q, pos);
    }
    for (int hh = 0; hh < nkv; hh++) {
      float *k = kv + (size_t)t * 2 * nkv * hd + (size_t)hh * hd;
      const float *v = kv + (size_t)t * 2 * nkv * hd + (size_t)(nkv + hh) * hd;
      rmsnorm(k, k, knw, 1, hd, c->eps);
      rope(m, k, pos);
      memcpy(kc + ((size_t)pos * nkv + hh) * hd, k, (size_t)hd * sizeof(float));
      memcpy(vc + ((size_t)pos * nkv + hh) * hd, v, (size_t)hd * sizeof(float));
    }
  }
  JD_PE();
  float *att = sc->mix; /* n x nh x hd */
  JD_PB(JD_P_ATTN);
  float scale = 1.0f / sqrtf((float)hd);
  float *p = sc->small; /* scores, length cap */
  for (int t = 0; t < n; t++) {
    int last = s->pos + t;
    for (int hh = 0; hh < nh; hh++) {
      const float *q = qg + (size_t)t * 2 * nh * hd + (size_t)hh * 2 * hd;
      int g = hh / rep;
      float mx = -INFINITY;
#if defined(__arm__) && defined(__ARM_FP)
      if (hd % 8 == 0) {
        for (int j = 0; j <= last; j += 2) {
          const float *k0 = kc + ((size_t)j * nkv + g) * hd;
          const float *k1 = j + 1 <= last ? k0 + (size_t)nkv * hd : k0;
          float d[2] = {0, 0};
          attn_qk2(q, k0, k1, hd, d);
          p[j] = d[0] * scale;
          if (p[j] > mx) mx = p[j];
          if (j + 1 <= last) {
            p[j + 1] = d[1] * scale;
            if (p[j + 1] > mx) mx = p[j + 1];
          }
        }
      } else
#endif
        for (int j = 0; j <= last; j++) {
          const float *k = kc + ((size_t)j * nkv + g) * hd;
          float d = 0;
          for (int i = 0; i < hd; i++) d += q[i] * k[i];
          p[j] = d * scale;
          if (p[j] > mx) mx = p[j];
        }
      float sum = 0;
      for (int j = 0; j <= last; j++) { p[j] = expf(p[j] - mx); sum += p[j]; }
      float inv = 1.0f / sum;
      float *o = att + ((size_t)t * nh + hh) * hd;
      for (int i = 0; i < hd; i++) o[i] = 0;
#if defined(__arm__) && defined(__ARM_FP)
      if (hd % 8 == 0) {
        for (int j = 0; j <= last; j++) p[j] = p[j] * inv; /* the weights, as below */
        for (int i = 0; i < hd; i += 8) attn_pv8(vc + (size_t)g * hd + i, nkv * hd, p, last + 1, o + i);
      } else
#endif
        for (int j = 0; j <= last; j++) {
          const float *v = vc + ((size_t)j * nkv + g) * hd;
          float w = p[j] * inv;
          for (int i = 0; i < hd; i++) o[i] += w * v[i];
        }
      const float *gate = qg + (size_t)t * 2 * nh * hd + (size_t)hh * 2 * hd + hd;
      for (int i = 0; i < hd; i++) o[i] *= sigmoidf_(gate[i]);
    }
  }
  JD_PE();
  JD_PB(JD_P_ATTN_OUT);
  jd_matmul(out, H, att, n, nh * hd, &L->o);
  JD_PE();
}

static void mlp(const jd_model *m, const jd_layer *L, const float *h, int n, float *out, scratch *sc) {
  int H = m->c.hidden, I = m->c.inter;
  float *g = sc->big1, *u = sc->big2;
  jd_matmul(g, I, h, n, H, &L->gate);
  jd_matmul(u, I, h, n, H, &L->up);
  for (size_t i = 0; i < (size_t)n * I; i++) g[i] = silu(g[i]) * u[i];
  jd_matmul(out, H, g, n, I, &L->down);
}

int jd_forward(const jd_model *m, jd_state *s, const int32_t *ids, int n, float *out, int chunk,
               jd_layer_dump_fn layer_dump, void *dump_ctx) {
  const jd_config *c = &m->c;
  if (s->pos + n > s->cap) return -1;
  if (chunk <= 0) chunk = 64;
  int H = c->hidden, cd = conv_dim(c);
  int wide = c->inter;
  if (cd > wide) wide = cd;
  if (2 * c->n_heads * c->head_dim > wide) wide = 2 * c->n_heads * c->head_dim;
  int mixw = cd > H ? cd : H;
  if (c->n_heads * c->head_dim > mixw) mixw = c->n_heads * c->head_dim;
  int smallw = s->cap > 2 * c->lin_nv ? s->cap : 2 * c->lin_nv;
  scratch sc;
  float *x = malloc((size_t)chunk * H * sizeof(float));
  sc.h = malloc((size_t)chunk * H * sizeof(float));
  sc.mix = malloc((size_t)chunk * mixw * sizeof(float));
  sc.big1 = malloc((size_t)chunk * wide * sizeof(float));
  sc.big2 = malloc((size_t)chunk * wide * sizeof(float));
  sc.small = malloc((size_t)(chunk > 1 ? chunk : 1) * smallw * sizeof(float));
  float *o = malloc((size_t)chunk * H * sizeof(float));
  int rc = -1;
  if (!x || !sc.h || !sc.mix || !sc.big1 || !sc.big2 || !sc.small || !o) goto done;
  for (int t0 = 0; t0 < n; t0 += chunk) {
    int nc = n - t0 < chunk ? n - t0 : chunk;
    JD_PB(JD_P_EMBED);
    embed_rows(m, ids + t0, nc, x);
    JD_PE();
    for (int li = 0; li < c->n_layers; li++) {
      const jd_layer *L = &m->layers[li];
      int lin = L->type == JD_LAYER_LINEAR;
      JD_PB(lin ? JD_P_LIN_PROJ : JD_P_ATTN_PROJ);
      rmsnorm(sc.h, x, (const float *)L->in_norm.data, nc, H, c->eps);
      JD_PE();
      if (lin) gated_deltanet(m, L, li, s, sc.h, nc, o, &sc);
      else attention(m, L, li, s, sc.h, nc, o, &sc);
      JD_PB(lin ? JD_P_LIN_OUT : JD_P_ATTN_OUT);
      for (size_t i = 0; i < (size_t)nc * H; i++) x[i] += o[i];
      JD_PE();
      JD_PB(JD_P_MLP);
      rmsnorm(sc.h, x, (const float *)L->post_norm.data, nc, H, c->eps);
      mlp(m, L, sc.h, nc, o, &sc);
      for (size_t i = 0; i < (size_t)nc * H; i++) x[i] += o[i];
      JD_PE();
      if (layer_dump) layer_dump(dump_ctx, li, s->pos, nc, x, H);
    }
    JD_PB(JD_P_HEAD);
    rmsnorm(out + (size_t)t0 * H, x, (const float *)m->final_norm.data, nc, H, c->eps);
    JD_PE();
    s->pos += nc;
  }
  rc = 0;
done:
  free(x); free(sc.h); free(sc.mix); free(sc.big1); free(sc.big2); free(sc.small); free(o);
  return rc;
}
