/* Byte-level BPE with the Qwen pre-tokenizer, loaded from a .jdt file (tools/export_jdw.py). */
#ifndef JD_TOKENIZER_H
#define JD_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
  uint32_t a, b, rank, result;
} jd_merge_slot;

typedef struct {
  uint32_t id;
  uint32_t len;
  const char *text;
} jd_added_token;

typedef struct {
  void *map;
  size_t map_len;
  uint32_t n_vocab;
  uint32_t hash_slots;
  const int32_t *byte_id;
  const uint32_t *vocab_off;
  const char *vocab_blob;
  const jd_merge_slot *hash;
  uint32_t n_added;
  jd_added_token *added;
  int marks; /* the split regex counts \p{M} with letters (Qwen3.5's own), not only \p{L} (Qwen2's) */
} jd_tokenizer;

typedef struct {
  int32_t *ids;
  uint32_t *start; /* byte offsets into the encoded text */
  uint32_t *end;
  size_t n, cap;
} jd_tokens;

int jd_tok_load(jd_tokenizer *t, const char *path);
void jd_tok_free(jd_tokenizer *t);

/* Encode UTF-8 text that is already NFC (the renderer normalises every input field), with no
 * special tokens added: the Qwen tokenizer adds none. Added tokens such as <|endoftext|> that
 * occur in the text are matched first, as Hugging Face tokenizers matches them. Appends to
 * `out`. Returns 0, or -1 on invalid UTF-8 or allocation failure. */
int jd_tok_encode(const jd_tokenizer *t, const char *text, size_t len, jd_tokens *out);

void jd_tokens_free(jd_tokens *t);

/* The bytes of one vocabulary entry (for debugging and tests). */
const char *jd_tok_piece(const jd_tokenizer *t, uint32_t id, uint32_t *len);

#endif
