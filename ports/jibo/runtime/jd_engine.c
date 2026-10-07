#include "jd_engine.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jd_kernels.h"
#include "jd_unicode.h"
#include "jd_json.h"

/* ---- strings ----------------------------------------------------------------------------- */

typedef struct {
  char *s;
  size_t n, cap;
} sbuf;

static int sb_put(sbuf *b, const char *s, size_t n) {
  if (b->n + n + 1 > b->cap) {
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->n + n + 1) cap *= 2;
    char *p = realloc(b->s, cap);
    if (!p) return -1;
    b->s = p;
    b->cap = cap;
  }
  memcpy(b->s + b->n, s, n);
  b->n += n;
  b->s[b->n] = 0;
  return 0;
}
static int sb_str(sbuf *b, const char *s) { return sb_put(b, s, strlen(s)); }
static int sb_fmt(sbuf *b, const char *fmt, ...) {
  char tmp[512];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
  va_end(ap);
  return n < 0 || (size_t)n >= sizeof(tmp) ? -1 : sb_put(b, tmp, (size_t)n);
}

static char *xstrdup(const char *s) {
  size_t n = strlen(s);
  char *r = malloc(n + 1);
  if (r) memcpy(r, s, n + 1);
  return r;
}

/* Length in bytes of the UTF-8 sequence starting at p. */
static int u8len(unsigned char c) { return c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4; }

static uint32_t u8cp(const char *p) {
  uint32_t cp;
  jd_utf8_decode(p, (size_t)u8len((unsigned char)*p), &cp);
  return cp;
}

/* Python's str.strip(): drop leading and trailing whitespace (str.isspace). */
static char *py_strip(const char *s) {
  size_t n = strlen(s), a = 0, b = n;
  while (a < n && jd_is_pyspace(u8cp(s + a))) a += (size_t)u8len((unsigned char)s[a]);
  while (b > a) {
    size_t k = b - 1;
    while (k > a && ((unsigned char)s[k] & 0xC0) == 0x80) k--;
    if (!jd_is_pyspace(u8cp(s + k))) break;
    b = k;
  }
  char *r = malloc(b - a + 1);
  if (!r) return NULL;
  memcpy(r, s + a, b - a);
  r[b - a] = 0;
  return r;
}

/* Python's " ".join(s.split()). */
static char *py_collapse(const char *s) {
  sbuf b = {0};
  size_t i = 0, n = strlen(s);
  int first = 1;
  if (sb_put(&b, "", 0) != 0) return NULL;
  while (i < n) {
    while (i < n && jd_is_pyspace(u8cp(s + i))) i += (size_t)u8len((unsigned char)s[i]);
    size_t start = i;
    while (i < n && !jd_is_pyspace(u8cp(s + i))) i += (size_t)u8len((unsigned char)s[i]);
    if (i > start) {
      if (!first && sb_put(&b, " ", 1) != 0) { free(b.s); return NULL; }
      if (sb_put(&b, s + start, i - start) != 0) { free(b.s); return NULL; }
      first = 0;
    }
  }
  return b.s;
}

static int put_jstr(sbuf *b, const char *s) {
  size_t len = 0, cap = 64;
  char *q = malloc(cap);
  if (!q) return -1;
  q[0] = 0;
  q = jd_json_quote(q, &len, &cap, s);
  int rc = q ? sb_put(b, q, len) : -1;
  free(q);
  return rc;
}

/* ---- rendering (prompting.py) ------------------------------------------------------------ */

typedef struct {
  char *text;
  int kind, n;
  char **labels, **descs;
  size_t *span_a, *span_b; /* byte spans of each option line within text */
} rendered;

static void rendered_free(rendered *r) {
  for (int i = 0; i < r->n; i++) {
    if (r->labels) free(r->labels[i]);
    if (r->descs) free(r->descs[i]);
  }
  free(r->labels); free(r->descs); free(r->span_a); free(r->span_b); free(r->text);
  memset(r, 0, sizeof(*r));
}

static const char *HEADERS[3] = {
  "Decide whether the statement is true of the state.",
  "Select exactly one option.",
  "Rate the state against the ordered levels below (lowest first).",
};
static const char *KINDS[3] = {"noul", "choice", "score"};

static int render_question(const jd_question *q, rendered *out) {
  memset(out, 0, sizeof(*out));
  out->kind = q->kind;
  int n = q->kind == JD_NOUL ? 2 : q->n_opts;
  out->n = n;
  out->labels = calloc((size_t)n, sizeof(char *));
  out->descs = calloc((size_t)n, sizeof(char *));
  out->span_a = calloc((size_t)n, sizeof(size_t));
  out->span_b = calloc((size_t)n, sizeof(size_t));
  char **pair_desc = calloc((size_t)n, sizeof(char *)); /* the description as rendered */
  sbuf b = {0};
  int rc = -1;
  char *instr = py_strip(q->instructions);
  if (!out->labels || !out->descs || !out->span_a || !out->span_b || !pair_desc || !instr) goto done;
  for (int i = 0; i < n; i++) {
    const char *label, *desc;
    char numbuf[16];
    if (q->kind == JD_NOUL) {
      static const char *DEFAULTS[2] = {"the statement does not hold for this state",
                                        "the statement holds for this state"};
      label = i == 0 ? "false" : "true";
      desc = DEFAULTS[i];
      for (int k = 0; k < q->n_opts; k++)
        if (strcmp(q->opt_names[k], label) == 0) desc = q->opt_descs[k] ? q->opt_descs[k] : "";
      pair_desc[i] = py_strip(desc);
    } else if (q->kind == JD_CHOICE) {
      label = q->opt_names[i];
      pair_desc[i] = py_strip(q->opt_descs[i] ? q->opt_descs[i] : "");
    } else {
      snprintf(numbuf, sizeof(numbuf), "%d", i);
      label = numbuf;
      pair_desc[i] = xstrdup(q->opt_descs[i]); /* score rubric text is not stripped first */
    }
    out->labels[i] = xstrdup(label);
    out->descs[i] = py_strip(pair_desc[i] ? pair_desc[i] : "");
    if (!pair_desc[i] || !out->labels[i] || !out->descs[i]) goto done;
  }
  if (sb_fmt(&b, "<question type=\"%s\">\n%s\n", KINDS[q->kind], HEADERS[q->kind]) || sb_str(&b, instr) ||
      sb_str(&b, "\n<options>\n"))
    goto done;
  for (int i = 0; i < n; i++) {
    char *d = py_collapse(pair_desc[i]);
    if (!d) goto done;
    out->span_a[i] = b.n;
    int bad = sb_fmt(&b, "%d. ", i + 1) || sb_str(&b, out->labels[i]) ||
              (*d && (sb_str(&b, " \xE2\x80\x94 ") || sb_str(&b, d)));
    free(d);
    if (bad) goto done;
    out->span_b[i] = b.n;
    if (i + 1 < n && sb_put(&b, "\n", 1)) goto done;
  }
  if (sb_str(&b, "\n</options>\n</question>\n<answer>")) goto done;
  out->text = b.s;
  b.s = NULL;
  rc = 0;
done:
  for (int i = 0; pair_desc && i < n; i++) free(pair_desc[i]);
  free(pair_desc);
  free(instr);
  free(b.s);
  if (rc) rendered_free(out);
  return rc;
}

/* ---- the engine -------------------------------------------------------------------------- */

int jd_engine_init(jd_engine *e, const jd_model *m, const jd_tokenizer *tok, jd_engine_opts o) {
  memset(e, 0, sizeof(*e));
  e->m = m;
  e->tok = tok;
  if (o.max_length <= 0) o.max_length = m->c.max_length;
  if (o.chunk <= 0) o.chunk = 64;
  e->o = o;
  e->hidden = malloc((size_t)o.max_length * m->c.hidden * sizeof(float));
  if (!e->hidden || jd_state_init(&e->s, m, o.max_length) || jd_state_init(&e->snap, m, o.max_length)) {
    jd_engine_free(e);
    return -1;
  }
  return 0;
}

void jd_engine_free(jd_engine *e) {
  if (e->m) {
    jd_state_free(&e->s, e->m);
    jd_state_free(&e->snap, e->m);
  }
  free(e->hidden);
  memset(e, 0, sizeof(*e));
}

void jd_answer_free(jd_answer *a) {
  for (int i = 0; i < a->n; i++) {
    if (a->labels) free(a->labels[i]);
    if (a->descs) free(a->descs[i]);
  }
  free(a->labels); free(a->descs); free(a->probs);
  memset(a, 0, sizeof(*a));
}

/* Last token of each option line (_option_token_index). -1 when truncation removed it. */
static int option_index(const jd_tokens *q, size_t cut, size_t a, size_t b) {
  int last = -1;
  for (size_t j = cut; j < q->n; j++) {
    if (q->end[j] <= q->start[j]) continue;
    if (q->start[j] >= a && q->end[j] <= b) last = (int)(j - cut);
  }
  return last;
}

static void layernorm(float *y, const float *x, const float *w, const float *b, int d, float eps) {
  float mean = 0, var = 0;
  for (int i = 0; i < d; i++) mean += x[i];
  mean /= (float)d;
  for (int i = 0; i < d; i++) { float t = x[i] - mean; var += t * t; }
  var /= (float)d;
  float inv = 1.0f / sqrtf(var + eps);
  for (int i = 0; i < d; i++) y[i] = (x[i] - mean) * inv * w[i] + b[i];
}

static void project(float *y, const jd_tensor *w, const jd_tensor *bias, const float *x, int in) {
  int out = (int)w->dims[0];
  const float *W = w->data, *B = bias->data;
  for (int o = 0; o < out; o++) {
    float s = 0;
    for (int i = 0; i < in; i++) s += W[(size_t)o * in + i] * x[i];
    y[o] = s + B[o];
  }
}

/* Pointer head: logits_k = <q(LN(decide)), k(LN(option_k))> / sqrt(dim), then / temperature,
 * then softmax over the options. */
static void head(const jd_model *m, const float *decide, const float *const *opts, int n, float temp,
                 float *probs) {
  int H = m->c.hidden, P = m->c.pointer_dim;
  float *nrm = malloc((size_t)H * sizeof(float)), *q = malloc((size_t)P * sizeof(float)),
        *k = malloc((size_t)P * sizeof(float));
  double *lg = malloc((size_t)n * sizeof(double));
  if (!nrm || !q || !k || !lg) { free(nrm); free(q); free(k); free(lg); return; }
  layernorm(nrm, decide, m->head_ln_w.data, m->head_ln_b.data, H, m->c.head_ln_eps);
  project(q, &m->head_q_w, &m->head_q_b, nrm, H);
  float scale = 1.0f / sqrtf((float)P);
  for (int j = 0; j < n; j++) {
    layernorm(nrm, opts[j], m->head_ln_w.data, m->head_ln_b.data, H, m->c.head_ln_eps);
    project(k, &m->head_k_w, &m->head_k_b, nrm, H);
    float d = 0;
    for (int i = 0; i < P; i++) d += k[i] * q[i];
    lg[j] = (double)((d * scale) / temp);
  }
  double mx = lg[0], sum = 0;
  for (int j = 1; j < n; j++) if (lg[j] > mx) mx = lg[j];
  for (int j = 0; j < n; j++) sum += exp(lg[j] - mx);
  for (int j = 0; j < n; j++) probs[j] = (float)exp(lg[j] - mx - log(sum));
  free(nrm); free(q); free(k); free(lg);
}

static float temp_for(const jd_model *m, int kind) {
  return kind == JD_NOUL ? m->c.temp_noul : kind == JD_CHOICE ? m->c.temp_choice : m->c.temp_score;
}

#define FAIL(...) do { snprintf(err, err_len, __VA_ARGS__); goto done; } while (0)

int jd_evaluate(jd_engine *e, const jd_request *r, jd_answer *answers, long *input_tokens, char *err,
                size_t err_len) {
  const jd_model *m = e->m;
  int H = m->c.hidden, max_len = e->o.max_length, rc = -1;
  int nq = r->n_q;
  rendered *rq = calloc((size_t)nq, sizeof(rendered));
  jd_tokens *qt = calloc((size_t)nq, sizeof(jd_tokens));
  jd_tokens st = {0};
  size_t *cut = calloc((size_t)nq, sizeof(size_t));
  char *state_text = NULL;
  int32_t *ids = NULL;
  memset(answers, 0, (size_t)nq * sizeof(jd_answer));
  *input_tokens = 0;
  if (!rq || !qt || !cut) FAIL("out of memory");

  for (int i = 0; i < nq; i++) {
    if (render_question(&r->q[i], &rq[i]) != 0) FAIL("cannot render question %s", r->q[i].name);
    if (jd_tok_encode(e->tok, rq[i].text, strlen(rq[i].text), &qt[i]) != 0) FAIL("cannot tokenise question");
  }
  {
    char *content = py_strip(r->state);
    sbuf b = {0};
    if (!content || sb_str(&b, "<state>\n") || sb_str(&b, content) || sb_str(&b, "\n</state>\n")) {
      free(content); free(b.s);
      FAIL("out of memory");
    }
    free(content);
    state_text = b.s;
  }
  if (jd_tok_encode(e->tok, state_text, strlen(state_text), &st) != 0) FAIL("cannot tokenise state");

  /* _fit: the question claims the window first; the state keeps its first tokens */
  size_t longest = 0;
  for (int i = 0; i < nq; i++) if (qt[i].n > longest) longest = qt[i].n;
  size_t reserve, budget;
  if (e->o.strict_window) {
    if (st.n + longest > (size_t)max_len)
      FAIL("prompt of %zu tokens exceeds the context window of %d tokens", st.n + longest, max_len);
    reserve = longest;
    budget = st.n;
  } else {
    size_t cap = (size_t)(max_len * 0.75);
    if (cap < 1) cap = 1;
    reserve = longest < cap ? longest : cap;
    budget = (size_t)max_len > reserve ? (size_t)max_len - reserve : 1;
  }
  for (int i = 0; i < nq; i++) cut[i] = qt[i].n > reserve ? qt[i].n - reserve : 0;
  size_t ns = st.n < budget ? st.n : budget;

  ids = malloc(((size_t)max_len + 1) * sizeof(int32_t));
  if (!ids) FAIL("out of memory");
  for (size_t j = 0; j < ns; j++) ids[j] = st.ids[j];

  int done_prefix = 0;
  for (int start = 0; start < nq; start += 32) { /* EngineConfig.max_batch */
    int end = start + 32 < nq ? start + 32 : nq;
    int shared = e->o.use_prefix_cache && end - start > 1;
    if (shared) {
      if (!done_prefix) {
        jd_state_reset(&e->snap, m);
        if (jd_forward(m, &e->snap, ids, (int)ns, e->hidden, e->o.chunk, NULL, NULL) != 0)
          FAIL("forward failed");
        done_prefix = 1;
      }
      *input_tokens += (long)ns;
    }
    for (int i = start; i < end; i++) {
      size_t nqi = qt[i].n - cut[i];
      size_t base = shared ? 0 : ns;
      if (shared) jd_state_copy(&e->s, &e->snap, m);
      else jd_state_reset(&e->s, m);
      for (size_t j = 0; j < nqi; j++) ids[base + j] = qt[i].ids[cut[i] + j];
      if (shared) {
        if (jd_forward(m, &e->s, ids, (int)nqi, e->hidden, e->o.chunk, NULL, NULL) != 0) FAIL("forward failed");
      } else {
        for (size_t j = 0; j < ns; j++) ids[j] = st.ids[j];
        if (jd_forward(m, &e->s, ids, (int)(ns + nqi), e->hidden, e->o.chunk, NULL, NULL) != 0)
          FAIL("forward failed");
      }
      *input_tokens += (long)(base + nqi);
      jd_answer *a = &answers[i];
      a->kind = rq[i].kind;
      a->n = rq[i].n;
      a->labels = rq[i].labels;
      a->descs = rq[i].descs;
      rq[i].labels = rq[i].descs = NULL;
      a->probs = calloc((size_t)a->n, sizeof(float));
      const float **opts = calloc((size_t)a->n, sizeof(float *));
      if (!a->probs || !opts) { free(opts); FAIL("out of memory"); }
      for (int k = 0; k < a->n; k++) {
        int idx = option_index(&qt[i], cut[i], rq[i].span_a[k], rq[i].span_b[k]);
        if (idx < 0) {
          free(opts);
          FAIL("option span (%zu,%zu) has no tokens left; the prompt was truncated through its option list",
               rq[i].span_a[k], rq[i].span_b[k]);
        }
        opts[k] = e->hidden + (base + (size_t)idx) * H;
      }
      head(m, e->hidden + (base + nqi - 1) * H, opts, a->n, temp_for(m, a->kind), a->probs);
      free(opts);
    }
  }
  rc = 0;
done:
  if (rc) for (int i = 0; i < nq; i++) jd_answer_free(&answers[i]);
  for (int i = 0; rq && i < nq; i++) { rendered_free(&rq[i]); jd_tokens_free(&qt[i]); }
  free(rq); free(qt); free(cut); free(ids); free(state_text);
  jd_tokens_free(&st);
  return rc;
}

/* ---- requests ---------------------------------------------------------------------------- */

void jd_request_free(jd_request *r) {
  for (int i = 0; r->q && i < r->n_q; i++) {
    jd_question *q = &r->q[i];
    for (int k = 0; k < q->n_opts; k++) {
      if (q->opt_names) free(q->opt_names[k]);
      if (q->opt_descs) free(q->opt_descs[k]);
    }
    free(q->opt_names); free(q->opt_descs); free(q->name); free(q->instructions);
  }
  free(r->q);
  free(r->state);
  memset(r, 0, sizeof(*r));
}

/* A Content value (render_content): a string as given, an object or a list as Python's
 * json.dumps(indent=2, ensure_ascii=False) writes it; NFC-normalised either way. */
static char *content(const jd_json *v, int allow_null, char *err, size_t err_len, const char *what) {
  char *text = NULL;
  if (v && v->type == JD_JSTRING) text = v->str;
  else if (v && (v->type == JD_JOBJECT || v->type == JD_JARRAY)) text = jd_json_dumps_py(v);
  else if (allow_null && v && v->type == JD_JNULL) return NULL;
  else {
    snprintf(err, err_len, "%s must be a string, an object or a list", what);
    return NULL;
  }
  if (!text) { snprintf(err, err_len, "out of memory"); return NULL; }
  size_t n;
  char *s = jd_nfc(text, strlen(text), &n);
  if (text != v->str) free(text);
  if (!s) snprintf(err, err_len, "%s is not valid UTF-8", what);
  return s;
}

#undef FAIL
#define FAIL(...) do { snprintf(err, err_len, __VA_ARGS__); goto fail; } while (0)

int jd_request_parse(const char *json, jd_request *r, char *err, size_t err_len) {
  memset(r, 0, sizeof(*r));
  jd_json *root = jd_json_parse(json, err, err_len);
  if (!root) goto fail;
  if (root->type != JD_JOBJECT) FAIL("request is not a JSON object");
  const jd_json *state = jd_json_get(root, "state");
  const jd_json *qs = jd_json_get(root, "questions");
  const jd_json *images = jd_json_get(root, "images");
  if (!state) FAIL("state is required");
  if (images && images->type == JD_JARRAY && images->n > 0) FAIL("this engine has no vision tower");
  if (!(r->state = content(state, 0, err, err_len, "state"))) goto fail;
  if (!qs || qs->type != JD_JOBJECT || qs->n < 1) FAIL("questions must be a non-empty object");
  r->n_q = qs->n;
  r->q = calloc((size_t)r->n_q, sizeof(jd_question));
  if (!r->q) FAIL("out of memory");
  for (int i = 0; i < qs->n; i++) {
    const jd_json *it = qs->items[i];
    jd_question *q = &r->q[i];
    if (!(q->name = xstrdup(qs->keys[i]))) FAIL("out of memory");
    if (it->type != JD_JOBJECT) FAIL("question %s is not an object", q->name);
    const jd_json *type = jd_json_get(it, "type");
    const jd_json *ins = jd_json_get(it, "instructions");
    const jd_json *crit = jd_json_get(it, "criteria");
    if (!type || type->type != JD_JSTRING) FAIL("question %s has no type", q->name);
    if (strcmp(type->str, "noul") == 0) q->kind = JD_NOUL;
    else if (strcmp(type->str, "choice") == 0) q->kind = JD_CHOICE;
    else if (strcmp(type->str, "score") == 0) q->kind = JD_SCORE;
    else FAIL("question %s: unknown type %s", q->name, type->str);
    if (!ins) FAIL("question %s has no instructions", q->name);
    if (!(q->instructions = content(ins, 0, err, err_len, "instructions"))) goto fail;
    if (q->kind == JD_SCORE) {
      if (!crit || crit->type != JD_JARRAY) FAIL("score criteria must be a list");
      q->n_opts = crit->n;
      if (q->n_opts < 2 || q->n_opts > 10) FAIL("score requires between 2 and 10 levels");
    } else if (crit && crit->type != JD_JNULL) {
      if (crit->type != JD_JOBJECT) FAIL("criteria must be an object");
      q->n_opts = crit->n;
    } else if (q->kind == JD_CHOICE) {
      FAIL("choice requires criteria");
    }
    if (q->kind == JD_CHOICE && (q->n_opts < 2 || q->n_opts > 255)) FAIL("choice requires 2 to 255 options");
    q->opt_names = calloc((size_t)q->n_opts + 1, sizeof(char *));
    q->opt_descs = calloc((size_t)q->n_opts + 1, sizeof(char *));
    if (!q->opt_names || !q->opt_descs) FAIL("out of memory");
    for (int k = 0; k < q->n_opts; k++) {
      const jd_json *o = crit->items[k];
      if (q->kind == JD_SCORE) {
        if (o->type != JD_JSTRING) FAIL("score levels must be strings");
        if (!(q->opt_descs[k] = content(o, 0, err, err_len, "level"))) goto fail;
        continue;
      }
      size_t n;
      const char *key = crit->keys[k];
      if (!(q->opt_names[k] = jd_nfc(key, strlen(key), &n))) FAIL("option name is not valid UTF-8");
      if (q->kind == JD_NOUL && strcmp(key, "true") && strcmp(key, "false"))
        FAIL("noul criteria keys must be a subset of {'true', 'false'}");
      err[0] = 0;
      q->opt_descs[k] = content(o, 1, err, err_len, "option description");
      if (!q->opt_descs[k] && err[0]) goto fail;
    }
  }
  jd_json_free(root);
  return 0;
fail:
  jd_json_free(root);
  jd_request_free(r);
  return -1;
}

char *jd_render_json(const jd_request *r) {
  sbuf b = {0};
  char *content = py_strip(r->state);
  int bad = !content || sb_str(&b, "[") || sb_str(&b, "") ;
  if (!bad) {
    sbuf st = {0};
    bad = sb_str(&st, "<state>\n") || sb_str(&st, content) || sb_str(&st, "\n</state>\n") || put_jstr(&b, st.s);
    free(st.s);
  }
  free(content);
  for (int i = 0; i < r->n_q && !bad; i++) {
    rendered rq;
    if (render_question(&r->q[i], &rq) != 0) { bad = 1; break; }
    bad = sb_str(&b, ",") || put_jstr(&b, rq.text);
    rendered_free(&rq);
  }
  if (bad || sb_str(&b, "]")) { free(b.s); return NULL; }
  return b.s;
}

/* ---- responses --------------------------------------------------------------------------- */

/* round(x, 4) as Python prints it: shortest form, at least one digit after the point. */
static int put_num(sbuf *b, double x) {
  char tmp[64];
  snprintf(tmp, sizeof(tmp), "%.4f", x);
  if (strcmp(tmp, "-0.0000") == 0) strcpy(tmp, "-0.0");
  size_t n = strlen(tmp);
  while (n > 0 && tmp[n - 1] == '0' && tmp[n - 2] != '.') tmp[--n] = 0;
  return sb_str(b, tmp);
}

char *jd_response_json(const jd_request *r, const jd_answer *ans, long input_tokens, const char *model_name,
                       float ordinal_smoothing) {
  sbuf b = {0};
  int bad = sb_str(&b, "{\"model\":") || put_jstr(&b, model_name) || sb_str(&b, ",\"answers\":{");
  for (int i = 0; i < r->n_q && !bad; i++) {
    const jd_answer *a = &ans[i];
    bad |= (i ? sb_str(&b, ",") : 0) || put_jstr(&b, r->q[i].name) || sb_str(&b, ":{\"type\":");
    bad |= put_jstr(&b, KINDS[a->kind]);
    if (a->kind == JD_NOUL) {
      int t = strcmp(a->labels[0], "true") == 0 ? 0 : 1;
      bad |= sb_str(&b, ",\"noul\":") || put_num(&b, (double)a->probs[t]);
    } else if (a->kind == JD_CHOICE) {
      int best = 0;
      double pmax = a->probs[0];
      for (int k = 1; k < a->n; k++) if ((double)a->probs[k] > pmax) { pmax = a->probs[k]; best = k; }
      double conf = a->n <= 1 ? 1.0 : (a->n * pmax - 1.0) / (a->n - 1.0);
      conf = conf < 0 ? 0 : conf > 1 ? 1 : conf;
      bad |= sb_str(&b, ",\"choice\":") || put_jstr(&b, a->labels[best]) || sb_str(&b, ",\"probabilities\":{");
      for (int k = 0; k < a->n && !bad; k++)
        bad |= (k ? sb_str(&b, ",") : 0) || put_jstr(&b, a->labels[k]) || sb_str(&b, ":") ||
               put_num(&b, (double)a->probs[k]);
      bad |= sb_str(&b, "},\"confidence\":") || put_num(&b, conf);
    } else {
      /* re-key by level, then expected value and the ordinal confidence (schema.py) */
      int n = a->n;
      double *p = calloc((size_t)n, sizeof(double));
      const char **legend = calloc((size_t)n, sizeof(char *));
      if (!p || !legend) { free(p); free(legend); bad = 1; break; }
      double expected = 0, total = 0;
      for (int k = 0; k < n; k++) {
        int lvl = atoi(a->labels[k]);
        p[lvl] = a->probs[k];
        legend[lvl] = a->descs[k];
        expected += lvl * (double)a->probs[k];
      }
      for (int k = 0; k < n; k++) total += p[k];
      double conf = 1.0;
      if (n > 1) {
        if (total <= 0) conf = 0.0;
        else {
          double mean = 0, var = 0;
          for (int k = 0; k < n; k++) mean += k * (p[k] / total);
          for (int k = 0; k < n; k++) var += (p[k] / total) * (k - mean) * (k - mean);
          double sigma = sqrt(var), smax = (n - 1) / 2.0;
          double sfloor = ordinal_smoothing > 0 ? sqrt((double)ordinal_smoothing) : 0.0;
          if (smax <= sfloor) conf = sigma <= sfloor ? 1.0 : 0.0;
          else conf = (smax - sigma) / (smax - sfloor);
          conf = conf < 0 ? 0 : conf > 1 ? 1 : conf;
        }
      }
      bad |= sb_str(&b, ",\"score\":") || put_num(&b, expected) || sb_str(&b, ",\"legend\":{");
      for (int k = 0; k < n && !bad; k++)
        bad |= (k ? sb_str(&b, ",") : 0) || sb_fmt(&b, "\"%d\":", k) || put_jstr(&b, legend[k]);
      bad |= sb_str(&b, "},\"probabilities\":{");
      for (int k = 0; k < n && !bad; k++)
        bad |= (k ? sb_str(&b, ",") : 0) || sb_fmt(&b, "\"%d\":", k) || put_num(&b, p[k]);
      bad |= sb_str(&b, "},\"confidence\":") || put_num(&b, conf);
      free(p);
      free(legend);
    }
    bad |= sb_str(&b, "}");
  }
  bad |= sb_str(&b, "},\"usage\":{\"input_tokens\":") || sb_fmt(&b, "%ld", input_tokens) ||
         sb_fmt(&b, ",\"output_tokens\":%d}}", r->n_q);
  if (bad) { free(b.s); return NULL; }
  return b.s;
}
