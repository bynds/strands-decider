/* Test helper: encode length-prefixed UTF-8 strings from stdin, one output line per string:
 * "id:start:end id:start:end ...". Strings are NFC-normalised first (as the renderer does)
 * when the second argument is "nfc"; with "nfc-only" it prints the normalised text's hex. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../runtime/jd_tokenizer.h"
#include "../runtime/jd_unicode.h"

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: tok_dump TOKENIZER.jdt [nfc|nfc-only] < strings\n"); return 2; }
  const char *mode = argc > 2 ? argv[2] : "";
  jd_tokenizer t;
  if (strcmp(mode, "nfc-only") != 0 && jd_tok_load(&t, argv[1]) != 0) { fprintf(stderr, "load failed\n"); return 1; }
  unsigned char lenb[4];
  while (fread(lenb, 1, 4, stdin) == 4) {
    size_t len = (size_t)lenb[0] | ((size_t)lenb[1] << 8) | ((size_t)lenb[2] << 16) | ((size_t)lenb[3] << 24);
    char *s = malloc(len + 1);
    if (!s || fread(s, 1, len, stdin) != len) return 1;
    s[len] = 0;
    char *text = s;
    size_t tlen = len;
    if (strncmp(mode, "nfc", 3) == 0) {
      text = jd_nfc(s, len, &tlen);
      if (!text) { printf("ERR\n"); free(s); continue; }
    }
    if (strcmp(mode, "nfc-only") == 0) {
      for (size_t i = 0; i < tlen; i++) printf("%02x", (unsigned char)text[i]);
      printf("\n");
    } else {
      jd_tokens out = {0};
      if (jd_tok_encode(&t, text, tlen, &out) != 0) printf("ERR\n");
      else {
        for (size_t i = 0; i < out.n; i++) printf(i ? " %d:%u:%u" : "%d:%u:%u", out.ids[i], out.start[i], out.end[i]);
        printf("\n");
      }
      jd_tokens_free(&out);
    }
    if (text != s) free(text);
    free(s);
  }
  return 0;
}
