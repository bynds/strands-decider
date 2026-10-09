/* jd_prof.h: opt-in per-operation cost accounting (-DJD_PROFILE) for perfvm/bench.sh. Each span
 * adds the advance of a caller-supplied clock (instructions retired in the perfvm guest,
 * nanoseconds on a host) to its operation. Without JD_PROFILE the macros are empty blocks and the
 * build is the shipped one.
 *
 *   JD_PB(JD_P_MLP); ...work... JD_PE();      spans do not nest; a goto out of one skips it */
#ifndef JD_PROF_H
#define JD_PROF_H

#include <stdint.h>

enum {
  JD_P_TOKENIZE = 0, /* rendering the prompt pieces and tokenising them */
  JD_P_EMBED,        /* embedding rows */
  JD_P_LIN_PROJ,     /* DeltaNet layers: input norm and the qkv, z, b, a projections */
  JD_P_LIN_CONV,     /* the causal depthwise convolution, SiLU, the conv state */
  JD_P_LIN_REC,      /* q/k norms, gates, the delta-rule recurrence, the gated RMSNorm */
  JD_P_LIN_OUT,      /* the output projection and the residual */
  JD_P_ATTN_PROJ,    /* attention layers: input norm and the q (with gate), k, v projections */
  JD_P_ATTN_ROPE,    /* q/k norms, RoPE, cache writes */
  JD_P_ATTN,         /* scores, softmax, weighted values, the output gate */
  JD_P_ATTN_OUT,     /* the o projection and the residual */
  JD_P_MLP,          /* post norm, gate and up, SiLU, down, the residual */
  JD_P_HEAD,         /* final norm and the pointer head */
  JD_P_STATE,        /* resetting and copying the recurrent and attention state */
  JD_P_N
};

extern const char *const jd_prof_names[JD_P_N];

/* Set the clock (NULL: spans record nothing); read and reset the totals. */
void jd_prof_set_clock(uint64_t (*clock)(void));
void jd_prof_take(uint64_t total[JD_P_N], uint64_t calls[JD_P_N]);

#ifdef JD_PROFILE
extern uint64_t (*jd_prof_clock)(void);
extern uint64_t jd_prof_total[JD_P_N], jd_prof_calls[JD_P_N];
#define JD_PB(op)                                             \
  {                                                           \
    const int jd_p_op = (op);                                 \
    const uint64_t jd_p_t0 = jd_prof_clock ? jd_prof_clock() : 0;
#define JD_PE()                                               \
  if (jd_prof_clock) {                                        \
    jd_prof_total[jd_p_op] += jd_prof_clock() - jd_p_t0;      \
    jd_prof_calls[jd_p_op]++;                                 \
  }                                                           \
  }
#else
#define JD_PB(op) {
#define JD_PE() }
#endif

#endif
