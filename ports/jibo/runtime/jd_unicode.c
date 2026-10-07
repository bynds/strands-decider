#include "jd_unicode.h"

#include <stdlib.h>
#include <string.h>

#include "unicode_tables.h"

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

long jd_utf8_decode(const char *s, size_t len, uint32_t *out) {
  const unsigned char *p = (const unsigned char *)s;
  size_t i = 0;
  long n = 0;
  while (i < len) {
    uint32_t c = p[i], cp;
    int extra;
    if (c < 0x80) { cp = c; extra = 0; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
    else return -1;
    for (int k = 1; k <= extra; k++) {
      if (i + (size_t)k >= len || (p[i + k] & 0xC0) != 0x80) return -1;
      cp = (cp << 6) | (p[i + k] & 0x3F);
    }
    if ((extra == 1 && cp < 0x80) || (extra == 2 && cp < 0x800) || (extra == 3 && cp < 0x10000))
      return -1; /* overlong */
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return -1;
    out[n++] = cp;
    i += (size_t)extra + 1;
  }
  return n;
}

int jd_utf8_encode(uint32_t cp, char *out) {
  unsigned char *o = (unsigned char *)out;
  if (cp < 0x80) { o[0] = (unsigned char)cp; return 1; }
  if (cp < 0x800) { o[0] = 0xC0 | (cp >> 6); o[1] = 0x80 | (cp & 0x3F); return 2; }
  if (cp < 0x10000) {
    o[0] = 0xE0 | (cp >> 12); o[1] = 0x80 | ((cp >> 6) & 0x3F); o[2] = 0x80 | (cp & 0x3F);
    return 3;
  }
  o[0] = 0xF0 | (cp >> 18); o[1] = 0x80 | ((cp >> 12) & 0x3F);
  o[2] = 0x80 | ((cp >> 6) & 0x3F); o[3] = 0x80 | (cp & 0x3F);
  return 4;
}

static int in_ranges(const jd_range *r, size_t n, uint32_t cp) {
  size_t lo = 0, hi = n;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (cp < r[mid].lo) hi = mid;
    else if (cp > r[mid].hi) lo = mid + 1;
    else return 1;
  }
  return 0;
}

int jd_is_letter(uint32_t cp) {
  if (cp < 0x80) return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
  return in_ranges(jd_letter_ranges, COUNT(jd_letter_ranges), cp);
}
int jd_is_mark(uint32_t cp) {
  if (cp < 0x300) return 0;
  return in_ranges(jd_mark_ranges, COUNT(jd_mark_ranges), cp);
}
int jd_is_number(uint32_t cp) {
  if (cp < 0x80) return cp >= '0' && cp <= '9';
  return in_ranges(jd_number_ranges, COUNT(jd_number_ranges), cp);
}
int jd_is_space(uint32_t cp) { return in_ranges(jd_space_ranges, COUNT(jd_space_ranges), cp); }
int jd_is_pyspace(uint32_t cp) { return in_ranges(jd_pyspace_ranges, COUNT(jd_pyspace_ranges), cp); }

/* ---- NFC ------------------------------------------------------------------------------ */

static uint8_t ccc_of(uint32_t cp) {
  size_t lo = 0, hi = COUNT(jd_ccc_table);
  if (cp < 0x300) return 0;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (cp < jd_ccc_table[mid].cp) hi = mid;
    else if (cp > jd_ccc_table[mid].cp) lo = mid + 1;
    else return jd_ccc_table[mid].ccc;
  }
  return 0;
}

static const jd_decomp *decomp_of(uint32_t cp) {
  size_t lo = 0, hi = COUNT(jd_decomp_table);
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (cp < jd_decomp_table[mid].cp) hi = mid;
    else if (cp > jd_decomp_table[mid].cp) lo = mid + 1;
    else return &jd_decomp_table[mid];
  }
  return NULL;
}

static uint32_t compose_pair(uint32_t a, uint32_t b) {
  /* Hangul LV and LVT */
  if (a >= JD_HANGUL_L_BASE && a < JD_HANGUL_L_BASE + JD_HANGUL_L_COUNT &&
      b >= JD_HANGUL_V_BASE && b < JD_HANGUL_V_BASE + JD_HANGUL_V_COUNT)
    return JD_HANGUL_S_BASE +
           ((a - JD_HANGUL_L_BASE) * JD_HANGUL_V_COUNT + (b - JD_HANGUL_V_BASE)) * JD_HANGUL_T_COUNT;
  if (a >= JD_HANGUL_S_BASE && a < JD_HANGUL_S_BASE + JD_HANGUL_S_COUNT &&
      (a - JD_HANGUL_S_BASE) % JD_HANGUL_T_COUNT == 0 &&
      b > JD_HANGUL_T_BASE && b < JD_HANGUL_T_BASE + JD_HANGUL_T_COUNT)
    return a + (b - JD_HANGUL_T_BASE);
  size_t lo = 0, hi = COUNT(jd_comp_table);
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    const jd_comp *c = &jd_comp_table[mid];
    if (a < c->a || (a == c->a && b < c->b)) hi = mid;
    else if (a > c->a || (a == c->a && b > c->b)) lo = mid + 1;
    else return c->c;
  }
  return 0;
}

/* Full canonical decomposition of one codepoint, appended to out. Returns the new length. */
static size_t decompose(uint32_t cp, uint32_t *out, size_t n) {
  if (cp >= JD_HANGUL_S_BASE && cp < JD_HANGUL_S_BASE + JD_HANGUL_S_COUNT) {
    uint32_t s = cp - JD_HANGUL_S_BASE;
    out[n++] = JD_HANGUL_L_BASE + s / (JD_HANGUL_V_COUNT * JD_HANGUL_T_COUNT);
    out[n++] = JD_HANGUL_V_BASE + (s % (JD_HANGUL_V_COUNT * JD_HANGUL_T_COUNT)) / JD_HANGUL_T_COUNT;
    if (s % JD_HANGUL_T_COUNT) out[n++] = JD_HANGUL_T_BASE + s % JD_HANGUL_T_COUNT;
    return n;
  }
  const jd_decomp *d = cp < 0xC0 ? NULL : decomp_of(cp);
  if (!d) { out[n++] = cp; return n; }
  n = decompose(d->a, out, n);
  if (d->b) n = decompose(d->b, out, n);
  return n;
}

char *jd_nfc(const char *s, size_t len, size_t *out_len) {
  /* Fast path: ASCII is already NFC. */
  size_t i;
  for (i = 0; i < len && (unsigned char)s[i] < 0x80; i++) {}
  if (i == len) {
    char *r = malloc(len + 1);
    if (!r) return NULL;
    memcpy(r, s, len);
    r[len] = 0;
    *out_len = len;
    return r;
  }
  uint32_t *cps = malloc(len * sizeof(uint32_t) + 4);
  /* A canonical decomposition is at most 4 codepoints long (3 for Hangul); 4x is safe. */
  uint32_t *dec = malloc(len * 4 * sizeof(uint32_t) + 16);
  char *res = NULL;
  if (!cps || !dec) goto done;
  long n = jd_utf8_decode(s, len, cps);
  if (n < 0) goto done;
  size_t m = 0;
  for (long k = 0; k < n; k++) m = decompose(cps[k], dec, m);
  /* Canonical ordering: stable insertion sort of each run of non-starters by class. */
  for (size_t k = 1; k < m; k++) {
    uint8_t c = ccc_of(dec[k]);
    if (!c) continue;
    size_t j = k;
    while (j > 0) {
      uint8_t p = ccc_of(dec[j - 1]);
      if (p <= c) break;
      uint32_t t = dec[j]; dec[j] = dec[j - 1]; dec[j - 1] = t;
      j--;
    }
  }
  /* Canonical composition. */
  if (m > 0) {
    size_t starter = 0, w = 1;
    int starter_valid = ccc_of(dec[0]) == 0;
    uint8_t last_cc = starter_valid ? 0 : 255;
    for (size_t k = 1; k < m; k++) {
      uint32_t c = dec[k];
      uint8_t cc = ccc_of(c);
      uint32_t comp = 0;
      if (starter_valid && (last_cc < cc || (last_cc == 0 && w == starter + 1)))
        comp = compose_pair(dec[starter], c);
      if (comp) {
        dec[starter] = comp;
        continue;
      }
      if (cc == 0) { starter = w; starter_valid = 1; }
      last_cc = cc;
      dec[w++] = c;
    }
    m = w;
  }
  res = malloc(m * 4 + 1);
  if (!res) goto done;
  size_t o = 0;
  for (size_t k = 0; k < m; k++) o += (size_t)jd_utf8_encode(dec[k], res + o);
  res[o] = 0;
  *out_len = o;
done:
  free(cps);
  free(dec);
  return res;
}
