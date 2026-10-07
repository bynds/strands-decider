#include "jd_tokenizer.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "jd_unicode.h"

#define JDT_MAGIC "JDT1"
#define JDT_HEADER 32

static uint32_t rd32(const unsigned char *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* Must match merge_hash() in tools/export_jdw.py. */
static uint32_t merge_hash(uint32_t a, uint32_t b, uint32_t slots) {
  return ((a * 0x9E3779B1u) ^ (b * 0x85EBCA77u)) & (slots - 1);
}

int jd_tok_load(jd_tokenizer *t, const char *path) {
  memset(t, 0, sizeof(*t));
  int fd = open(path, O_RDONLY);
  if (fd < 0) return -1;
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < JDT_HEADER + 1024) { close(fd); return -1; }
  void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (m == MAP_FAILED) return -1;
  t->map = m;
  t->map_len = (size_t)st.st_size;
  const unsigned char *p = m;
  if (memcmp(p, JDT_MAGIC, 4) != 0 || rd32(p + 4) != 1) goto fail;
  t->n_vocab = rd32(p + 8);
  uint32_t blob_len = rd32(p + 12);
  t->hash_slots = rd32(p + 16);
  t->n_added = rd32(p + 20);
  t->marks = (int)(rd32(p + 24) & 1);
  if (t->hash_slots == 0 || (t->hash_slots & (t->hash_slots - 1))) goto fail;
  size_t off = JDT_HEADER;
  t->byte_id = (const int32_t *)(p + off);
  off += 256 * 4;
  t->vocab_off = (const uint32_t *)(p + off);
  off += ((size_t)t->n_vocab + 1) * 4;
  t->vocab_blob = (const char *)(p + off);
  off += (blob_len + 15u) & ~15u;
  t->hash = (const jd_merge_slot *)(p + off);
  off += (size_t)t->hash_slots * sizeof(jd_merge_slot);
  if (off > t->map_len) goto fail;
  t->added = calloc(t->n_added ? t->n_added : 1, sizeof(jd_added_token));
  if (!t->added) goto fail;
  for (uint32_t i = 0; i < t->n_added; i++) {
    if (off + 12 > t->map_len) goto fail;
    t->added[i].id = rd32(p + off);
    t->added[i].len = rd32(p + off + 4);
    /* p + off + 8: flags (special, normalized, lstrip, rstrip); the exporter refuses any
     * added token that is normalized or strips, so matching is exact. */
    t->added[i].text = (const char *)(p + off + 12);
    off += 12 + ((t->added[i].len + 3u) & ~3u);
    if (off > t->map_len) goto fail;
  }
  return 0;
fail:
  jd_tok_free(t);
  return -1;
}

void jd_tok_free(jd_tokenizer *t) {
  if (t->map) munmap(t->map, t->map_len);
  free(t->added);
  memset(t, 0, sizeof(*t));
}

void jd_tokens_free(jd_tokens *t) {
  free(t->ids);
  free(t->start);
  free(t->end);
  memset(t, 0, sizeof(*t));
}

const char *jd_tok_piece(const jd_tokenizer *t, uint32_t id, uint32_t *len) {
  if (id >= t->n_vocab) { *len = 0; return ""; }
  *len = t->vocab_off[id + 1] - t->vocab_off[id];
  return t->vocab_blob + t->vocab_off[id];
}

static int push(jd_tokens *o, int32_t id, uint32_t a, uint32_t b) {
  if (o->n == o->cap) {
    size_t cap = o->cap ? o->cap * 2 : 256;
    int32_t *ids = realloc(o->ids, cap * sizeof(int32_t));
    if (!ids) return -1;
    o->ids = ids;
    uint32_t *s = realloc(o->start, cap * sizeof(uint32_t));
    if (!s) return -1;
    o->start = s;
    uint32_t *e = realloc(o->end, cap * sizeof(uint32_t));
    if (!e) return -1;
    o->end = e;
    o->cap = cap;
  }
  o->ids[o->n] = id;
  o->start[o->n] = a;
  o->end[o->n] = b;
  o->n++;
  return 0;
}

static const jd_merge_slot *find_merge(const jd_tokenizer *t, uint32_t a, uint32_t b) {
  uint32_t h = merge_hash(a, b, t->hash_slots);
  for (;;) {
    const jd_merge_slot *s = &t->hash[h];
    if (s->a == 0xFFFFFFFFu) return NULL;
    if (s->a == a && s->b == b) return s;
    h = (h + 1) & (t->hash_slots - 1);
  }
}

/* BPE over the bytes [a, b) of text: repeatedly merge the adjacent pair of lowest rank,
 * the leftmost on a tie, as Hugging Face's BPE model does. */
static int bpe_piece(const jd_tokenizer *t, const char *text, uint32_t a, uint32_t b, jd_tokens *o) {
  uint32_t n = b - a;
  int32_t stack_ids[64];
  uint32_t stack_lo[64];
  int32_t *ids = n <= 64 ? stack_ids : malloc(n * sizeof(int32_t));
  uint32_t *lo = n <= 64 ? stack_lo : malloc((n + 1) * sizeof(uint32_t));
  if (!ids || !lo) {
    if (ids != stack_ids) free(ids);
    if (lo != stack_lo) free(lo);
    return -1;
  }
  for (uint32_t i = 0; i < n; i++) {
    ids[i] = t->byte_id[(unsigned char)text[a + i]];
    lo[i] = a + i;
  }
  uint32_t m = n;
  while (m > 1) {
    uint32_t best = 0xFFFFFFFFu, at = 0, result = 0;
    for (uint32_t i = 0; i + 1 < m; i++) {
      const jd_merge_slot *s = find_merge(t, (uint32_t)ids[i], (uint32_t)ids[i + 1]);
      if (s && s->rank < best) { best = s->rank; at = i; result = s->result; }
    }
    if (best == 0xFFFFFFFFu) break;
    ids[at] = (int32_t)result;
    memmove(ids + at + 1, ids + at + 2, (m - at - 2) * sizeof(int32_t));
    memmove(lo + at + 1, lo + at + 2, (m - at - 2) * sizeof(uint32_t));
    m--;
  }
  int rc = 0;
  for (uint32_t i = 0; i < m && rc == 0; i++)
    rc = push(o, ids[i], lo[i], i + 1 < m ? lo[i + 1] : b);
  if (ids != stack_ids) free(ids);
  if (lo != stack_lo) free(lo);
  return rc;
}

/* ---- the pre-tokenizer ------------------------------------------------------------------
 * Qwen2's split regex, which the Qwen2Tokenizer class applies:
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|
 *   \s*[\r\n]+|\s+(?!\S)|\s+
 * and, with `marks`, Qwen3.5's, where \p{L}+ becomes [\p{L}\p{M}]+ and the symbol class also
 * excludes \p{M}. Each alternative below returns the length (in codepoints) it matches at i, or
 * 0. They are tried in order at each position, as the regex engine tries them; together they
 * match every codepoint, so the pieces cover the text. */

static int is_crlf(uint32_t c) { return c == '\r' || c == '\n'; }
static uint32_t lower_ascii(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

static size_t alt_contraction(const uint32_t *c, size_t n, size_t i) {
  if (c[i] != '\'' || i + 1 >= n) return 0;
  uint32_t x = lower_ascii(c[i + 1]);
  if (c[i + 1] == 0x17F) x = 's'; /* LATIN SMALL LETTER LONG S case-folds to s */
  if (x == 's' || x == 't') return 2;
  if (i + 2 < n) {
    uint32_t y = lower_ascii(c[i + 2]);
    if ((x == 'r' && y == 'e') || (x == 'v' && y == 'e')) return 3;
  }
  if (x == 'm') return 2;
  if (i + 2 < n && x == 'l' && lower_ascii(c[i + 2]) == 'l') return 3;
  if (x == 'd') return 2;
  return 0;
}

static int is_wordch(uint32_t c, int marks) { return jd_is_letter(c) || (marks && jd_is_mark(c)); }

static size_t word_run(const uint32_t *c, size_t n, size_t i, int marks) {
  size_t k = i;
  while (k < n && is_wordch(c[k], marks)) k++;
  return k - i;
}

static size_t alt_word(const uint32_t *c, size_t n, size_t i, int marks) {
  if (!is_crlf(c[i]) && !jd_is_letter(c[i]) && !jd_is_number(c[i])) {
    size_t r = i + 1 < n ? word_run(c, n, i + 1, marks) : 0;
    if (r) return r + 1;
    /* without the optional prefix: only a mark (with `marks`) can start the run here */
  }
  return word_run(c, n, i, marks);
}

static int is_symbol(uint32_t c, int marks) {
  return !jd_is_space(c) && !jd_is_letter(c) && !jd_is_number(c) && !(marks && jd_is_mark(c));
}

static size_t alt_symbols(const uint32_t *c, size_t n, size_t i, int marks) {
  size_t k = i;
  if (c[k] == ' ') k++;
  size_t s = k;
  while (k < n && is_symbol(c[k], marks)) k++;
  if (k == s) return 0; /* without the space, c[i] == ' ' is whitespace: no match either */
  while (k < n && is_crlf(c[k])) k++;
  return k - i;
}

static size_t space_run(const uint32_t *c, size_t n, size_t i) {
  size_t k = i;
  while (k < n && jd_is_space(c[k])) k++;
  return k - i;
}

static size_t pretok_match(const uint32_t *c, size_t n, size_t i, int marks) {
  size_t r;
  if ((r = alt_contraction(c, n, i))) return r;
  if ((r = alt_word(c, n, i, marks))) return r;
  if (jd_is_number(c[i])) return 1;
  if ((r = alt_symbols(c, n, i, marks))) return r;
  size_t w = space_run(c, n, i);
  if (w) {
    /* \s*[\r\n]+ : up to and including the last CR or LF of the run */
    for (size_t k = w; k > 0; k--)
      if (is_crlf(c[i + k - 1])) return k;
    /* \s+(?!\S) : the whole run at the end of text, else all but its last character */
    if (i + w == n) return w;
    if (w >= 2) return w - 1;
    return w; /* \s+ */
  }
  return 1; /* unreachable: the alternatives cover every codepoint */
}

static int encode_segment(const jd_tokenizer *t, const char *text, size_t a, size_t b,
                          jd_tokens *o) {
  size_t len = b - a;
  if (len == 0) return 0;
  uint32_t *cps = malloc((len + 1) * sizeof(uint32_t));
  uint32_t *boff = malloc((len + 1) * sizeof(uint32_t));
  int rc = -1;
  if (!cps || !boff) goto done;
  long n = jd_utf8_decode(text + a, len, cps);
  if (n < 0) goto done;
  /* byte offset of each codepoint */
  {
    size_t p = a;
    for (long k = 0; k < n; k++) {
      boff[k] = (uint32_t)p;
      unsigned char c0 = (unsigned char)text[p];
      p += c0 < 0x80 ? 1 : c0 < 0xE0 ? 2 : c0 < 0xF0 ? 3 : 4;
    }
    boff[n] = (uint32_t)b;
  }
  for (size_t i = 0; i < (size_t)n;) {
    size_t r = pretok_match(cps, (size_t)n, i, t->marks);
    if (bpe_piece(t, text, boff[i], boff[i + r], o) != 0) goto done;
    i += r;
  }
  rc = 0;
done:
  free(cps);
  free(boff);
  return rc;
}

int jd_tok_encode(const jd_tokenizer *t, const char *text, size_t len, jd_tokens *out) {
  size_t seg = 0, i = 0;
  while (i < len) {
    /* leftmost-longest added token at i */
    const jd_added_token *best = NULL;
    if (text[i] == '<') {
      for (uint32_t k = 0; k < t->n_added; k++) {
        const jd_added_token *a = &t->added[k];
        if (a->len <= len - i && memcmp(text + i, a->text, a->len) == 0 &&
            (!best || a->len > best->len))
          best = a;
      }
    }
    if (!best) { i++; continue; }
    if (encode_segment(t, text, seg, i, out) != 0) return -1;
    if (push(out, (int32_t)best->id, (uint32_t)i, (uint32_t)(i + best->len)) != 0) return -1;
    i += best->len;
    seg = i;
  }
  return encode_segment(t, text, seg, len, out);
}
