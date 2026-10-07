/* jibo-decider: the decider on the CPU.
 *
 *   jibo-decider ask   MODEL.jdw TOKENIZER.jdt [options] < request.json
 *   jibo-decider serve MODEL.jdw TOKENIZER.jdt --socket PATH [options]
 *
 * `ask` reads one /v1/systemone request and prints the response, as `strands-decider serve` would
 * return it, with latency_ms. `serve` answers the same requests on a Unix socket, one per
 * connection: the client writes the request, shuts down its write side, and reads the response.
 * A request of {"health": true} returns the model's description instead. Requests are answered one
 * at a time, in order; the listen backlog is the queue.
 *
 * Options:
 *   --window N           context window in tokens (default: the checkpoint's max_length)
 *   --strict-window      refuse a prompt longer than the window instead of shortening it
 *   --no-prefix-cache    encode every question with its own copy of the state
 *   --chunk N            tokens per forward chunk (default 64)
 *   --raw                (ask) also print each question's unrounded probabilities, one JSON line each
 *   --model-name NAME    the "model" field of the response (default: the .jdw file name)
 *   --socket PATH        (serve) where to listen; created with mode 0660
 *   --max-temp C         (serve) refuse requests while any thermal zone named by --thermal is above C
 *   --thermal PATH       (serve, repeatable) a sysfs temperature file in millidegrees
 *   --min-avail-mb N     (serve) refuse requests while MemAvailable is below N MB
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "jd_engine.h"

#define MAX_THERMAL 8

typedef struct {
  jd_engine e;
  jd_model m;
  jd_tokenizer tok;
  const char *name;
  double max_temp;
  const char *thermal[MAX_THERMAL];
  int n_thermal;
  long min_avail_mb;
} server;

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static char *read_fd(int fd) {
  size_t cap = 1 << 16, n = 0;
  char *b = malloc(cap);
  if (!b) return NULL;
  for (;;) {
    if (n + 4096 > cap) {
      char *p = realloc(b, cap *= 2);
      if (!p) { free(b); return NULL; }
      b = p;
    }
    ssize_t r = read(fd, b + n, cap - n - 1);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) break;
    n += (size_t)r;
    if (n > (64u << 20)) break; /* no request is this large */
  }
  b[n] = 0;
  return b;
}

static int write_all(int fd, const char *s, size_t n) {
  while (n) {
    ssize_t w = write(fd, s, n);
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) return -1;
    s += w;
    n -= (size_t)w;
  }
  return 0;
}

/* {"detail": msg}, the shape FastAPI gives an HTTP 422 */
static char *detail(const char *msg) {
  size_t n = strlen(msg);
  char *o = malloc(n * 6 + 16), *p = o;
  if (!o) return NULL;
  p += sprintf(p, "{\"detail\":\"");
  for (const unsigned char *c = (const unsigned char *)msg; *c; c++) {
    if (*c == '"' || *c == '\\') { *p++ = '\\'; *p++ = (char)*c; }
    else if (*c < 0x20) p += sprintf(p, "\\u%04x", *c);
    else *p++ = (char)*c;
  }
  strcpy(p, "\"}");
  return o;
}

static char *handle(server *s, const char *json, int raw) {
  char err[512] = {0};
  jd_request r;
  if (jd_request_parse(json, &r, err, sizeof(err)) != 0) return detail(err);
  jd_answer *ans = calloc((size_t)r.n_q, sizeof(jd_answer));
  long ntok = 0;
  double t0 = now_ms();
  if (!ans || jd_evaluate(&s->e, &r, ans, &ntok, err, sizeof(err)) != 0) {
    free(ans);
    jd_request_free(&r);
    return detail(err[0] ? err : "out of memory");
  }
  double ms = now_ms() - t0;
  char *resp = jd_response_json(&r, ans, ntok, s->name, s->m.c.ordinal_smoothing);
  char *out = NULL;
  if (resp) {
    size_t L = strlen(resp), extra = 64;
    if (raw)
      for (int i = 0; i < r.n_q; i++) extra += 64 + strlen(r.q[i].name) + (size_t)ans[i].n * 48;
    for (int i = 0; raw && i < r.n_q; i++)
      for (int k = 0; k < ans[i].n; k++) extra += strlen(ans[i].labels[k]);
    out = malloc(L + extra);
    if (out) {
      char *p = out + sprintf(out, "%.*s,\"latency_ms\":%.2f}", (int)(L - 1), resp, ms);
      for (int i = 0; raw && i < r.n_q; i++) {
        p += sprintf(p, "\n{\"name\":\"%s\",\"labels\":[", r.q[i].name);
        for (int k = 0; k < ans[i].n; k++) p += sprintf(p, k ? ",\"%s\"" : "\"%s\"", ans[i].labels[k]);
        p += sprintf(p, "],\"probs\":[");
        for (int k = 0; k < ans[i].n; k++) p += sprintf(p, k ? ",%.9g" : "%.9g", (double)ans[i].probs[k]);
        p += sprintf(p, "]}");
      }
    }
    free(resp);
  }
  for (int i = 0; i < r.n_q; i++) jd_answer_free(&ans[i]);
  free(ans);
  jd_request_free(&r);
  return out;
}

/* The governor: a reason to refuse work now, or NULL. */
static const char *refuse(const server *s, char *buf, size_t n) {
  for (int i = 0; i < s->n_thermal; i++) {
    FILE *f = fopen(s->thermal[i], "r");
    long mc = 0;
    if (f && fscanf(f, "%ld", &mc) == 1 && mc / 1000.0 > s->max_temp) {
      fclose(f);
      snprintf(buf, n, "busy: %s reads %.1f C, above %.1f C", s->thermal[i], mc / 1000.0, s->max_temp);
      return buf;
    }
    if (f) fclose(f);
  }
  if (s->min_avail_mb > 0) {
    FILE *f = fopen("/proc/meminfo", "r");
    char line[256];
    long kb = -1;
    while (f && fgets(line, sizeof(line), f))
      if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
    if (f) fclose(f);
    if (kb >= 0 && kb / 1024 < s->min_avail_mb) {
      snprintf(buf, n, "busy: MemAvailable %ld MB, below %ld MB", kb / 1024, s->min_avail_mb);
      return buf;
    }
  }
  return NULL;
}

static int serve(server *s, const char *path) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  struct sockaddr_un addr;
  if (fd < 0 || strlen(path) >= sizeof(addr.sun_path)) { perror("socket"); return 1; }
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strcpy(addr.sun_path, path);
  unlink(path);
  if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0 || chmod(path, 0660) != 0 || listen(fd, 1) != 0) {
    perror("bind/listen");
    return 1;
  }
  signal(SIGPIPE, SIG_IGN);
  fprintf(stderr, "[jibo-decider] serving %s on %s\n", s->name, path);
  for (;;) {
    int c = accept(fd, NULL, NULL);
    if (c < 0) {
      if (errno == EINTR) continue;
      perror("accept");
      break;
    }
    char *req = read_fd(c), *resp = NULL, why[256];
    const char *no = refuse(s, why, sizeof(why));
    if (!req) resp = detail("cannot read the request");
    else if (strstr(req, "\"health\"") && !strstr(req, "\"questions\"")) {
      const char *bm = jd_model_meta(&s->m, "base_model");
      int bl = bm ? (int)strcspn(bm, "\n") : 0;
      resp = malloc(512 + strlen(s->name) + (size_t)bl);
      if (resp)
        sprintf(resp, "{\"status\":\"ok\",\"model\":\"%s\",\"base_model\":\"%.*s\",\"max_length\":%d,"
                      "\"temperature\":%.6g,\"device\":\"cpu\",\"prefix_cache\":%s}",
                s->name, bl, bm ? bm : "", s->e.o.max_length, s->m.c.temperature,
                s->e.o.use_prefix_cache ? "true" : "false");
    } else if (no) resp = detail(no);
    else resp = handle(s, req, 0);
    if (!resp) resp = detail("out of memory");
    if (resp) { write_all(c, resp, strlen(resp)); write_all(c, "\n", 1); }
    free(resp);
    free(req);
    close(c);
  }
  close(fd);
  return 1;
}

int main(int argc, char **argv) {
  int is_ask = argc >= 2 && !strcmp(argv[1], "ask"), is_serve = argc >= 2 && !strcmp(argv[1], "serve");
  if (argc < 4 || (!is_ask && !is_serve)) {
    fprintf(stderr, "usage: jibo-decider ask|serve MODEL.jdw TOKENIZER.jdt [options]\n");
    return 2;
  }
  static server s;
  jd_engine_opts o = {0, 0, 1, 64};
  int raw = 0;
  const char *sock = NULL;
  s.name = strrchr(argv[2], '/') ? strrchr(argv[2], '/') + 1 : argv[2];
  s.max_temp = 1e9;
  for (int i = 4; i < argc; i++) {
    if (!strcmp(argv[i], "--window") && i + 1 < argc) o.max_length = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--strict-window")) o.strict_window = 1;
    else if (!strcmp(argv[i], "--no-prefix-cache")) o.use_prefix_cache = 0;
    else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) o.chunk = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--raw")) raw = 1;
    else if (!strcmp(argv[i], "--model-name") && i + 1 < argc) s.name = argv[++i];
    else if (!strcmp(argv[i], "--socket") && i + 1 < argc) sock = argv[++i];
    else if (!strcmp(argv[i], "--max-temp") && i + 1 < argc) s.max_temp = atof(argv[++i]);
    else if (!strcmp(argv[i], "--thermal") && i + 1 < argc && s.n_thermal < MAX_THERMAL) s.thermal[s.n_thermal++] = argv[++i];
    else if (!strcmp(argv[i], "--min-avail-mb") && i + 1 < argc) s.min_avail_mb = atol(argv[++i]);
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
  }
  if (is_serve && !sock) { fprintf(stderr, "serve needs --socket PATH\n"); return 2; }
  if (jd_model_load(&s.m, argv[2]) != 0) { fprintf(stderr, "cannot load model %s\n", argv[2]); return 1; }
  if (jd_tok_load(&s.tok, argv[3]) != 0) { fprintf(stderr, "cannot load tokenizer %s\n", argv[3]); return 1; }
  if (jd_engine_init(&s.e, &s.m, &s.tok, o) != 0) { fprintf(stderr, "out of memory\n"); return 1; }
  if (is_serve) return serve(&s, sock);

  char *json = read_fd(0);
  char *resp = json ? handle(&s, json, raw) : NULL;
  int ok = resp && strncmp(resp, "{\"detail\"", 9) != 0;
  if (resp) puts(resp);
  free(resp);
  free(json);
  jd_engine_free(&s.e);
  jd_tok_free(&s.tok);
  jd_model_free(&s.m);
  return ok ? 0 : 3;
}
