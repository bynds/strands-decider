/* jd_icount.h: user-space instructions retired by this process, from the PMU through
 * perf_event_open. Under perfvm (qemu-system with -icount) the count is exact and repeatable; on a
 * host without a PMU jd_icount_open fails and callers fall back to time. Header-only. */
#ifndef JD_ICOUNT_H
#define JD_ICOUNT_H
#include <linux/perf_event.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

/* syscall(2) is not declared under strict -D_POSIX_C_SOURCE. */
long syscall(long number, ...);

static int jd_icount_fd = -1;

static int jd_icount_open(void) {
  struct perf_event_attr a;
  memset(&a, 0, sizeof a);
  a.type = PERF_TYPE_HARDWARE;
  a.size = sizeof a;
  a.config = PERF_COUNT_HW_INSTRUCTIONS;
  a.exclude_kernel = 1;
  a.exclude_hv = 1;
  jd_icount_fd = (int)syscall(__NR_perf_event_open, &a, 0, -1, -1, 0);
  return jd_icount_fd >= 0;
}

static uint64_t jd_icount_read(void) {
  uint64_t v = 0;
  if (jd_icount_fd < 0 || read(jd_icount_fd, &v, sizeof v) != (ssize_t)sizeof v) return 0;
  return v;
}
#endif
