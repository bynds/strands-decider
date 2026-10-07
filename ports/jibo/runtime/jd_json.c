#include "jd_json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jd_unicode.h"

#define MAX_DEPTH 500

typedef struct {
  const char *p;
  char *err;
  size_t err_len;
  int depth;
} parser;

static jd_json *fail(parser *ps, const char *msg) {
  if (ps->err[0] == 0) snprintf(ps->err, ps->err_len, "invalid JSON: %s", msg);
  return NULL;
}

static void skip_ws(parser *ps) {
  while (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r') ps->p++;
}

void jd_json_free(jd_json *j) {
  if (!j) return;
  for (int i = 0; i < j->n; i++) {
    jd_json_free(j->items[i]);
    if (j->keys) free(j->keys[i]);
  }
  free(j->items);
  free(j->keys);
  free(j->str);
  free(j);
}

static int hex4(const char *s, unsigned *out) {
  unsigned v = 0;
  for (int i = 0; i < 4; i++) {
    char c = s[i];
    v <<= 4;
    if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
    else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
    else return -1;
  }
  *out = v;
  return 0;
}

static char *parse_string(parser *ps) {
  if (*ps->p != '"') return NULL;
  ps->p++;
  size_t cap = 64, n = 0;
  char *o = malloc(cap);
  if (!o) return NULL;
  for (;;) {
    unsigned char c = (unsigned char)*ps->p;
    if (n + 8 > cap) {
      char *q = realloc(o, cap *= 2);
      if (!q) { free(o); return NULL; }
      o = q;
    }
    if (c == 0) { free(o); fail(ps, "unterminated string"); return NULL; }
    if (c < 0x20) { free(o); fail(ps, "control character in a string"); return NULL; }
    if (c == '"') { ps->p++; break; }
    if (c != '\\') { o[n++] = (char)c; ps->p++; continue; }
    char e = ps->p[1];
    ps->p += 2;
    switch (e) {
      case '"': o[n++] = '"'; break;
      case '\\': o[n++] = '\\'; break;
      case '/': o[n++] = '/'; break;
      case 'b': o[n++] = '\b'; break;
      case 'f': o[n++] = '\f'; break;
      case 'n': o[n++] = '\n'; break;
      case 'r': o[n++] = '\r'; break;
      case 't': o[n++] = '\t'; break;
      case 'u': {
        unsigned u, lo;
        if (hex4(ps->p, &u)) { free(o); fail(ps, "bad \\u escape"); return NULL; }
        ps->p += 4;
        if (u >= 0xD800 && u <= 0xDBFF) {
          if (ps->p[0] != '\\' || ps->p[1] != 'u' || hex4(ps->p + 2, &lo) || lo < 0xDC00 || lo > 0xDFFF) {
            free(o); fail(ps, "lone surrogate in a string"); return NULL;
          }
          ps->p += 6;
          u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
        } else if (u >= 0xDC00 && u <= 0xDFFF) {
          free(o); fail(ps, "lone surrogate in a string"); return NULL;
        }
        if (u == 0) { free(o); fail(ps, "\\u0000 is not supported"); return NULL; }
        n += (size_t)jd_utf8_encode(u, o + n);
        break;
      }
      default: free(o); fail(ps, "bad escape"); return NULL;
    }
  }
  o[n] = 0;
  /* the raw bytes must be valid UTF-8 too */
  uint32_t *tmp = malloc((n + 1) * sizeof(uint32_t));
  long ok = tmp ? jd_utf8_decode(o, n, tmp) : -1;
  free(tmp);
  if (ok < 0) { free(o); fail(ps, "a string is not valid UTF-8"); return NULL; }
  return o;
}

static jd_json *parse_value(parser *ps);

static jd_json *new_node(void) { return calloc(1, sizeof(jd_json)); }

static int push(jd_json *j, char *key, jd_json *v) {
  /* a repeated key keeps its first position and takes the new value */
  if (key) {
    for (int i = 0; i < j->n; i++) {
      if (strcmp(j->keys[i], key) == 0) {
        jd_json_free(j->items[i]);
        j->items[i] = v;
        free(key);
        return 0;
      }
    }
  }
  jd_json **it = realloc(j->items, sizeof(jd_json *) * (size_t)(j->n + 1));
  if (!it) return -1;
  j->items = it;
  if (key || j->keys) {
    char **k = realloc(j->keys, sizeof(char *) * (size_t)(j->n + 1));
    if (!k) return -1;
    j->keys = k;
    j->keys[j->n] = key;
  }
  j->items[j->n++] = v;
  return 0;
}

static jd_json *parse_value(parser *ps) {
  skip_ws(ps);
  if (++ps->depth > MAX_DEPTH) return fail(ps, "nested too deeply");
  jd_json *j = NULL;
  const char *p = ps->p;
  if (*p == '{' || *p == '[') {
    int obj = *p == '{';
    j = new_node();
    if (!j) return fail(ps, "out of memory");
    j->type = obj ? JD_JOBJECT : JD_JARRAY;
    if (obj) { j->keys = malloc(sizeof(char *)); if (!j->keys) { jd_json_free(j); return fail(ps, "out of memory"); } }
    ps->p++;
    skip_ws(ps);
    if (*ps->p == (obj ? '}' : ']')) { ps->p++; ps->depth--; return j; }
    for (;;) {
      char *key = NULL;
      if (obj) {
        skip_ws(ps);
        key = parse_string(ps);
        if (!key) { jd_json_free(j); return fail(ps, "expected a key"); }
        skip_ws(ps);
        if (*ps->p != ':') { free(key); jd_json_free(j); return fail(ps, "expected ':'"); }
        ps->p++;
      }
      jd_json *v = parse_value(ps);
      if (!v || push(j, key, v) != 0) { free(key); jd_json_free(v); jd_json_free(j); return fail(ps, "bad value"); }
      skip_ws(ps);
      if (*ps->p == ',') { ps->p++; continue; }
      if (*ps->p == (obj ? '}' : ']')) { ps->p++; break; }
      jd_json_free(j);
      return fail(ps, "expected ',' or a closing bracket");
    }
  } else if (*p == '"') {
    j = new_node();
    if (!j) return fail(ps, "out of memory");
    j->type = JD_JSTRING;
    if (!(j->str = parse_string(ps))) { jd_json_free(j); return fail(ps, "bad string"); }
  } else if (!strncmp(p, "true", 4) || !strncmp(p, "false", 5) || !strncmp(p, "null", 4)) {
    j = new_node();
    if (!j) return fail(ps, "out of memory");
    j->type = *p == 't' ? JD_JTRUE : *p == 'f' ? JD_JFALSE : JD_JNULL;
    ps->p += *p == 'f' ? 5 : 4;
  } else if (*p == '-' || (*p >= '0' && *p <= '9')) {
    const char *s = p;
    if (*p == '-') p++;
    if (*p == '0') p++;
    else if (*p >= '1' && *p <= '9') while (*p >= '0' && *p <= '9') p++;
    else return fail(ps, "bad number");
    if (*p == '.') { p++; if (!(*p >= '0' && *p <= '9')) return fail(ps, "bad number"); while (*p >= '0' && *p <= '9') p++; }
    if (*p == 'e' || *p == 'E') {
      p++;
      if (*p == '+' || *p == '-') p++;
      if (!(*p >= '0' && *p <= '9')) return fail(ps, "bad number");
      while (*p >= '0' && *p <= '9') p++;
    }
    j = new_node();
    if (!j || !(j->str = malloc((size_t)(p - s) + 1))) { free(j); return fail(ps, "out of memory"); }
    j->type = JD_JNUMBER;
    memcpy(j->str, s, (size_t)(p - s));
    j->str[p - s] = 0;
    ps->p = p;
  } else {
    return fail(ps, "unexpected character");
  }
  ps->depth--;
  return j;
}

jd_json *jd_json_parse(const char *text, char *err, size_t err_len) {
  parser ps = {text, err, err_len, 0};
  err[0] = 0;
  jd_json *j = parse_value(&ps);
  if (!j) return NULL;
  skip_ws(&ps);
  if (*ps.p) { jd_json_free(j); fail(&ps, "trailing characters"); return NULL; }
  return j;
}

const jd_json *jd_json_get(const jd_json *obj, const char *key) {
  if (!obj || obj->type != JD_JOBJECT) return NULL;
  for (int i = 0; i < obj->n; i++)
    if (strcmp(obj->keys[i], key) == 0) return obj->items[i];
  return NULL;
}

/* ---- writing ------------------------------------------------------------------------------- */

static char *put(char *b, size_t *len, size_t *cap, const char *s, size_t n) {
  if (!b) return NULL;
  if (*len + n + 1 > *cap) {
    size_t c = *cap ? *cap : 256;
    while (c < *len + n + 1) c *= 2;
    char *q = realloc(b, c);
    if (!q) { free(b); return NULL; }
    b = q;
    *cap = c;
  }
  memcpy(b + *len, s, n);
  *len += n;
  b[*len] = 0;
  return b;
}

char *jd_json_quote(char *b, size_t *len, size_t *cap, const char *s) {
  b = put(b, len, cap, "\"", 1);
  for (const unsigned char *c = (const unsigned char *)s; b && *c; c++) {
    char esc[8];
    const char *e = NULL;
    switch (*c) {
      case '"': e = "\\\""; break;
      case '\\': e = "\\\\"; break;
      case '\n': e = "\\n"; break;
      case '\r': e = "\\r"; break;
      case '\t': e = "\\t"; break;
      case '\b': e = "\\b"; break;
      case '\f': e = "\\f"; break;
      default:
        if (*c < 0x20) { snprintf(esc, sizeof(esc), "\\u%04x", *c); e = esc; }
    }
    b = e ? put(b, len, cap, e, strlen(e)) : put(b, len, cap, (const char *)c, 1);
  }
  return put(b, len, cap, "\"", 1);
}

/* Python's repr() of the float this JSON number denotes: the shortest digits that round-trip,
 * fixed notation for exponents -4 to 15 (with ".0" for whole numbers), otherwise d.ddde+XX. */
static void py_float_repr(double x, char *out) {
  if (x == 0) { strcpy(out, signbit(x) ? "-0.0" : "0.0"); return; }
  if (isinf(x)) { strcpy(out, x > 0 ? "Infinity" : "-Infinity"); return; }
  char buf[40];
  int prec;
  for (prec = 1; prec <= 17; prec++) {
    snprintf(buf, sizeof(buf), "%.*e", prec - 1, x);
    if (strtod(buf, NULL) == x) break;
  }
  /* buf is [-]d[.ddd]e[+-]XX: take the digits and the exponent */
  char digits[24];
  int nd = 0, neg = buf[0] == '-';
  const char *p = buf + neg;
  for (; *p && *p != 'e'; p++) if (*p != '.') digits[nd++] = *p;
  digits[nd] = 0;
  int exp = atoi(p + 1);
  char *o = out;
  if (neg) *o++ = '-';
  if (exp >= -4 && exp < 16) {
    if (exp < 0) {
      o += sprintf(o, "0.");
      for (int i = 0; i < -exp - 1; i++) *o++ = '0';
      o += sprintf(o, "%s", digits);
    } else {
      for (int i = 0; i <= exp; i++) *o++ = i < nd ? digits[i] : '0';
      *o++ = '.';
      if (nd > exp + 1) o += sprintf(o, "%s", digits + exp + 1);
      else *o++ = '0';
    }
    *o = 0;
  } else {
    *o++ = digits[0];
    if (nd > 1) { *o++ = '.'; o += sprintf(o, "%s", digits + 1); }
    sprintf(o, "e%c%02d", exp < 0 ? '-' : '+', exp < 0 ? -exp : exp);
  }
}

static char *dumps(char *b, size_t *len, size_t *cap, const jd_json *v, int level) {
  char tmp[64];
  switch (v->type) {
    case JD_JNULL: return put(b, len, cap, "null", 4);
    case JD_JTRUE: return put(b, len, cap, "true", 4);
    case JD_JFALSE: return put(b, len, cap, "false", 5);
    case JD_JSTRING: return jd_json_quote(b, len, cap, v->str);
    case JD_JNUMBER:
      if (strpbrk(v->str, ".eE")) {
        py_float_repr(strtod(v->str, NULL), tmp);
        return put(b, len, cap, tmp, strlen(tmp));
      }
      /* an integer: Python prints it as written, except -0 */
      if (!strcmp(v->str, "-0")) return put(b, len, cap, "0", 1);
      return put(b, len, cap, v->str, strlen(v->str));
    default: break;
  }
  int obj = v->type == JD_JOBJECT;
  if (v->n == 0) return put(b, len, cap, obj ? "{}" : "[]", 2);
  b = put(b, len, cap, obj ? "{" : "[", 1);
  for (int i = 0; i < v->n && b; i++) {
    b = put(b, len, cap, i ? ",\n" : "\n", i ? 2 : 1);
    for (int k = 0; k < (level + 1) * 2 && b; k++) b = put(b, len, cap, " ", 1);
    if (obj && b) { b = jd_json_quote(b, len, cap, v->keys[i]); b = put(b, len, cap, ": ", 2); }
    if (b) b = dumps(b, len, cap, v->items[i], level + 1);
  }
  b = put(b, len, cap, "\n", 1);
  for (int k = 0; k < level * 2 && b; k++) b = put(b, len, cap, " ", 1);
  return put(b, len, cap, obj ? "}" : "]", 1);
}

char *jd_json_dumps_py(const jd_json *v) {
  size_t len = 0, cap = 256;
  char *b = malloc(cap);
  if (!b) return NULL;
  b[0] = 0;
  return dumps(b, &len, &cap, v, 0);
}
