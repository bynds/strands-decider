/* selftest.c: checks the counter is exact. A loop of known shape must cost a fixed number of
 * instructions per iteration, the same on every run. */
#include <stdio.h>
#include "jd_icount.h"
int main(void) {
  volatile unsigned x = 0;
  unsigned i;
  uint64_t a, b, c;
  if (!jd_icount_open()) {
    perror("perf_event_open");
    return 1;
  }
  a = jd_icount_read();
  for (i = 0; i < 1000000; i++) x += i;
  b = jd_icount_read();
  for (i = 0; i < 2000000; i++) x += i;
  c = jd_icount_read();
  printf("selftest: 1M iters %llu instructions, 2M iters %llu (ratio %.6f)\n", (unsigned long long)(b - a),
         (unsigned long long)(c - b), (double)(c - b) / (double)(b - a));
  return 0;
}
