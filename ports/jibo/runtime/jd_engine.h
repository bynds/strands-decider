/* One request, N typed questions about one state: infer.py's SystemOneEngine.evaluate in C. */
#ifndef JD_ENGINE_H
#define JD_ENGINE_H

#include "jd_model.h"
#include "jd_tokenizer.h"

enum { JD_NOUL = 0, JD_CHOICE = 1, JD_SCORE = 2 };

typedef struct {
  char *name;          /* the key in the request */
  int kind;
  char *instructions;
  int n_opts;          /* choice: options; score: levels; noul: 0 or the criteria given */
  char **opt_names;    /* choice: option names; noul: "true"/"false" keys; score: unused */
  char **opt_descs;    /* descriptions; NULL entries for a null description */
} jd_question;

typedef struct {
  char *state;
  int n_q;
  jd_question *q;
} jd_request;

typedef struct {
  int kind;
  int n;               /* options */
  char **labels;       /* slot labels in rendered order */
  char **descs;        /* slot descriptions (stripped) in rendered order */
  float *probs;        /* in rendered order */
} jd_answer;

typedef struct {
  int max_length;      /* the window; the checkpoint's own unless overridden */
  int strict_window;
  int use_prefix_cache;
  int chunk;
} jd_engine_opts;

typedef struct {
  const jd_model *m;
  const jd_tokenizer *tok;
  jd_engine_opts o;
  jd_state s, snap;    /* working state and the shared-prefix snapshot */
  float *hidden;       /* max_length x hidden */
} jd_engine;

int jd_engine_init(jd_engine *e, const jd_model *m, const jd_tokenizer *tok, jd_engine_opts o);
void jd_engine_free(jd_engine *e);

/* Evaluate a request. answers must hold r->n_q entries; free them with jd_answer_free.
 * *input_tokens receives the token count the Python engine reports. Returns 0, or -1 with
 * an error message in err. */
int jd_evaluate(jd_engine *e, const jd_request *r, jd_answer *answers, long *input_tokens,
                char *err, size_t err_len);
void jd_answer_free(jd_answer *a);

/* Parse a /v1/systemone request (JSON). Returns 0, or -1 with a message in err. */
int jd_request_parse(const char *json, jd_request *r, char *err, size_t err_len);
void jd_request_free(jd_request *r);

/* The response JSON, as the Python server returns it (without latency_ms). malloc'd. */
char *jd_response_json(const jd_request *r, const jd_answer *answers, long input_tokens,
                       const char *model_name, float ordinal_smoothing);

#endif
