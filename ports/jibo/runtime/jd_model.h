/* The Qwen3.5 text torso on the CPU: weights from a .jdw file, one sequence at a time. */
#ifndef JD_MODEL_H
#define JD_MODEL_H

#include <stddef.h>
#include <stdint.h>

enum { JD_F32 = 0, JD_BF16 = 1, JD_Q8 = 3, JD_Q4 = 4 };

typedef struct {
  int dtype;
  int ndim;
  uint32_t dims[4];
  const void *data;
  uint64_t nbytes;
} jd_tensor;

enum { JD_LAYER_LINEAR = 0, JD_LAYER_ATTN = 1 };

typedef struct {
  int type;
  jd_tensor in_norm, post_norm, gate, up, down;
  /* Gated DeltaNet */
  jd_tensor qkv, z, b, a, conv, dt_bias, a_log, gnorm, out;
  /* gated attention */
  jd_tensor q, k, v, o, q_norm, k_norm;
} jd_layer;

typedef struct {
  int vocab, hidden, inter, n_layers;
  int n_heads, n_kv, head_dim, rot_dim;
  float rope_theta, eps;
  int lin_nk, lin_nv, lin_dk, lin_dv, conv_k;
  int max_length, pointer_dim;
  float head_ln_eps;
  float temperature, temp_noul, temp_choice, temp_score, ordinal_smoothing;
} jd_config;

typedef struct {
  void *map;
  size_t map_len;
  jd_config c;
  jd_tensor embed, final_norm;
  jd_tensor head_ln_w, head_ln_b, head_q_w, head_q_b, head_k_w, head_k_b;
  jd_layer *layers;
  char *meta; /* the file's key=value lines, NUL-terminated */
  float *rope_inv_freq;
  uint32_t n_tensors;
  size_t table_off;
} jd_model;

/* Recurrent and attention state of one sequence. */
typedef struct {
  int pos;          /* tokens consumed */
  int cap;          /* attention cache capacity in tokens */
  float **conv;     /* per layer (DeltaNet only): (conv_k - 1) x conv_dim, oldest first */
  float **rec;      /* per layer (DeltaNet only): nv x dk x dv */
  float **kc, **vc; /* per layer (attention only): cap x n_kv x head_dim */
} jd_state;

int jd_model_load(jd_model *m, const char *path);
void jd_model_free(jd_model *m);
const char *jd_model_meta(const jd_model *m, const char *key); /* NULL if absent; points into m->meta */
int jd_model_find(const jd_model *m, const char *name, jd_tensor *out); /* 0, or -1 if absent */

int jd_state_init(jd_state *s, const jd_model *m, int cap);
void jd_state_free(jd_state *s, const jd_model *m);
void jd_state_reset(jd_state *s, const jd_model *m);
void jd_state_copy(jd_state *dst, const jd_state *src, const jd_model *m);

/* Run `n` tokens through the torso, continuing `s`. Writes the final (normed) hidden state of
 * every token to `out` (n x hidden). Processes in chunks of `chunk` tokens. Returns 0, or -1
 * when the cache would overflow or memory runs out. If `layer_dump` is set, it is called with
 * each layer's output hidden states (for parity tests). */
typedef void (*jd_layer_dump_fn)(void *ctx, int layer, int t0, int n, const float *h, int hidden);
int jd_forward(const jd_model *m, jd_state *s, const int32_t *ids, int n, float *out, int chunk,
               jd_layer_dump_fn layer_dump, void *dump_ctx);

#endif
