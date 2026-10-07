/* A small JSON reader for requests, and Python's json.dumps(indent=2, ensure_ascii=False).
 *
 * The prompt of a structured state is the state as Python re-serialises it, so rendering must
 * match json.dumps character for character: key order kept, Python's float repr, integers as
 * written, and only quotes, backslashes and control characters escaped. Objects keep the first
 * position of a repeated key with its last value, as a Python dict built by json.loads does. */
#ifndef JD_JSON_H
#define JD_JSON_H

#include <stddef.h>

enum { JD_JNULL, JD_JTRUE, JD_JFALSE, JD_JNUMBER, JD_JSTRING, JD_JARRAY, JD_JOBJECT };

typedef struct jd_json {
  int type;
  char *str;              /* JD_JSTRING: the decoded UTF-8; JD_JNUMBER: the number as written */
  int n;                  /* items in an array or object */
  struct jd_json **items; /* values */
  char **keys;            /* object keys (decoded UTF-8) */
} jd_json;

/* Parse; NULL with a message in err on failure (as Python's json.loads would refuse, plus
 * \u0000 in a string, which the C strings of the runtime cannot hold). */
jd_json *jd_json_parse(const char *text, char *err, size_t err_len);
void jd_json_free(jd_json *j);
const jd_json *jd_json_get(const jd_json *obj, const char *key); /* NULL if absent */

/* json.dumps(v, indent=2, ensure_ascii=False), malloc'd. */
char *jd_json_dumps_py(const jd_json *v);

/* A JSON string literal for s (ensure_ascii=False escaping), appended to a malloc'd buffer:
 * returns the new buffer (or NULL on failure; the old one is then freed). */
char *jd_json_quote(char *buf, size_t *len, size_t *cap, const char *s);

#endif
