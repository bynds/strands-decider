#include "jd_prof.h"

#include <string.h>

const char *const jd_prof_names[JD_P_N] = {"tokenize", "embed", "lin_proj", "lin_conv", "lin_rec",
                                           "lin_out", "attn_proj", "attn_rope", "attn", "attn_out",
                                           "mlp", "head", "state"};

uint64_t (*jd_prof_clock)(void);
uint64_t jd_prof_total[JD_P_N], jd_prof_calls[JD_P_N];

void jd_prof_set_clock(uint64_t (*clock)(void)) { jd_prof_clock = clock; }

void jd_prof_take(uint64_t total[JD_P_N], uint64_t calls[JD_P_N]) {
  memcpy(total, jd_prof_total, sizeof(jd_prof_total));
  memcpy(calls, jd_prof_calls, sizeof(jd_prof_calls));
  memset(jd_prof_total, 0, sizeof(jd_prof_total));
  memset(jd_prof_calls, 0, sizeof(jd_prof_calls));
}
