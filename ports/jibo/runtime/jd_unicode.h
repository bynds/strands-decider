/* UTF-8, the tokenizer's character classes, and NFC. */
#ifndef JD_UNICODE_H
#define JD_UNICODE_H

#include <stddef.h>
#include <stdint.h>

/* Decode UTF-8 into codepoints. Returns the count, or -1 on invalid UTF-8 (overlong forms,
 * surrogates and values past U+10FFFF included). `out` needs room for `len` codepoints. */
long jd_utf8_decode(const char *s, size_t len, uint32_t *out);

/* Encode one codepoint; returns its byte length (1-4). `out` needs 4 bytes. */
int jd_utf8_encode(uint32_t cp, char *out);

/* The classes of the pre-tokenizer's regex, as the tokenizer's own engine defines them. */
int jd_is_letter(uint32_t cp);  /* \p{L} */
int jd_is_mark(uint32_t cp);    /* \p{M} */
int jd_is_number(uint32_t cp);  /* \p{N} */
int jd_is_space(uint32_t cp);   /* \s    */

/* Python's str.isspace(), which str.strip() and str.split() use when a prompt is rendered. */
int jd_is_pyspace(uint32_t cp);

/* NFC-normalise UTF-8 text. Returns a malloc'd NUL-terminated string and sets *out_len,
 * or NULL on invalid UTF-8 or allocation failure. */
char *jd_nfc(const char *s, size_t len, size_t *out_len);

#endif
