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
  if (c->lin_nv % c->lin_nk || c->n_heads % c->n_kv || c->rot_dim % 2 || c->conv_k < 2) goto fail;
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
  s->pos = 0;
  for (int i = 0; i < c->n_layers; i++) {
    if (s->conv[i]) memset(s->conv[i], 0, (size_t)(c->conv_k - 1) * conv_dim(c) * sizeof(float));
    if (s->rec[i]) memset(s->rec[i], 0, (size_t)c->lin_nv * c->lin_dk * c->lin_dv * sizeof(float));
  }
}

void jd_state_copy(jd_state *dst, const jd_state *src, const jd_model *m) {
  const jd_config *c = &m->c;
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

static void gated_deltanet(const jd_model *m, const jd_layer *L, int li, jd_state *s, const float *h,
                           int n, float *out, scratch *sc) {
  const jd_config *c = &m->c;
  int H = c->hidden, kd = c->lin_nk * c->lin_dk, vd = c->lin_nv * c->lin_dv, cd = conv_dim(c);
  int K = c->conv_k, nv = c->lin_nv, dk = c->lin_dk, dv = c->lin_dv, rep = c->lin_nv / c->lin_nk;
  float *mixed = sc->big1;   /* n x cd */
  float *z = sc->big2;       /* n x vd */
  float *ab = sc->small;     /* n x 2nv: b then a */
  jd_matmul(mixed, cd, h, n, H, &L->qkv);
  jd_matmul(z, vd, h, n, H, &L->z);
  jd_matmul(ab, 2 * nv, h, n, H, &L->b);
  jd_matmul(ab + nv, 2 * nv, h, n, H, &L->a);

  /* causal depthwise convolution over [state; mixed], then SiLU; the state keeps the last K-1 inputs */
  const float *w = (const float *)L->conv.data; /* cd x K */
  float *cs = s->conv[li];                      /* (K-1) x cd */
  float *conv_out = sc->mix;                    /* n x cd (mix is n x max(H, cd)) */
  for (int t = 0; t < n; t++) {
    for (int ch = 0; ch < cd; ch++) {
      float acc = 0.0f;
      for (int j = 0; j < K; j++) {
        int src = t - (K - 1) + j; /* index into mixed; negative reads the state */
        float v = src >= 0 ? mixed[(size_t)src * cd + ch] : cs[(size_t)(K - 1 + src) * cd + ch];
        acc += w[(size_t)ch * K + j] * v;
      }
      conv_out[(size_t)t * cd + ch] = silu(acc);
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

  const float *dt_bias = (const float *)L->dt_bias.data, *a_log = (const float *)L->a_log.data;
  const float *gw = (const float *)L->gnorm.data;
  float *S = s->rec[li];
  float qn[512], kn[512], kv_mem[512], o[512];
  float qscale = 1.0f / sqrtf((float)dk);
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
      /* gated RMSNorm over the head, then into the output row */
      float ss = 0;
      for (int j = 0; j < dv; j++) ss += o[j] * o[j];
      float inv = 1.0f / sqrtf(ss / (float)dv + c->eps);
      const float *zh = z + (size_t)t * vd + (size_t)hv * dv;
      for (int j = 0; j < dv; j++) o[j] = gw[j] * (o[j] * inv) * silu(zh[j]);
      memcpy(z + (size_t)t * vd + (size_t)hv * dv, o, (size_t)dv * sizeof(float));
    }
  }
  jd_matmul(out, H, z, n, vd, &L->out);
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

static void attention(const jd_model *m, const jd_layer *L, int li, jd_state *s, const float *h,
                      int n, float *out, scratch *sc) {
  const jd_config *c = &m->c;
  int H = c->hidden, nh = c->n_heads, nkv = c->n_kv, hd = c->head_dim, rep = nh / nkv;
  float *qg = sc->big1;  /* n x (2 nh hd): per head, query then gate */
  float *kv = sc->big2;  /* n x (2 nkv hd): keys then values */
  jd_matmul(qg, 2 * nh * hd, h, n, H, &L->q);
  jd_matmul(kv, 2 * nkv * hd, h, n, H, &L->k);
  jd_matmul(kv + nkv * hd, 2 * nkv * hd, h, n, H, &L->v);
  const float *qnw = (const float *)L->q_norm.data, *knw = (const float *)L->k_norm.data;
  float *kc = s->kc[li], *vc = s->vc[li];
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
  float *att = sc->mix; /* n x nh x hd */
  float scale = 1.0f / sqrtf((float)hd);
  float *p = sc->small; /* scores, length cap */
  for (int t = 0; t < n; t++) {
    int last = s->pos + t;
    for (int hh = 0; hh < nh; hh++) {
      const float *q = qg + (size_t)t * 2 * nh * hd + (size_t)hh * 2 * hd;
      int g = hh / rep;
      float mx = -INFINITY;
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
      for (int j = 0; j <= last; j++) {
        const float *v = vc + ((size_t)j * nkv + g) * hd;
        float w = p[j] * inv;
        for (int i = 0; i < hd; i++) o[i] += w * v[i];
      }
      const float *gate = qg + (size_t)t * 2 * nh * hd + (size_t)hh * 2 * hd + hd;
      for (int i = 0; i < hd; i++) o[i] *= sigmoidf_(gate[i]);
    }
  }
  jd_matmul(out, H, att, n, nh * hd, &L->o);
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
    embed_rows(m, ids + t0, nc, x);
    for (int li = 0; li < c->n_layers; li++) {
      const jd_layer *L = &m->layers[li];
      rmsnorm(sc.h, x, (const float *)L->in_norm.data, nc, H, c->eps);
      if (L->type == JD_LAYER_LINEAR) gated_deltanet(m, L, li, s, sc.h, nc, o, &sc);
      else attention(m, L, li, s, sc.h, nc, o, &sc);
      for (size_t i = 0; i < (size_t)nc * H; i++) x[i] += o[i];
      rmsnorm(sc.h, x, (const float *)L->post_norm.data, nc, H, c->eps);
      mlp(m, L, sc.h, nc, o, &sc);
      for (size_t i = 0; i < (size_t)nc * H; i++) x[i] += o[i];
      if (layer_dump) layer_dump(dump_ctx, li, s->pos, nc, x, H);
    }
    rmsnorm(out + (size_t)t0 * H, x, (const float *)m->final_norm.data, nc, H, c->eps);
    s->pos += nc;
  }
  rc = 0;
done:
  free(x); free(sc.h); free(sc.mix); free(sc.big1); free(sc.big2); free(sc.small); free(o);
  return rc;
}
