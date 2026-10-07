#define _POSIX_C_SOURCE 200809L
#include "jd_gl.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned int GLenum, GLuint, GLbitfield;
typedef int GLint, GLsizei;
typedef char GLchar;
typedef unsigned char GLubyte;
typedef intptr_t GLintptr, GLsizeiptr;
typedef uint64_t GLuint64;
typedef struct __GLsync *GLsync;
typedef struct _XDisplay Display;
typedef struct __GLXcontextRec *GLXContext;
typedef struct __GLXFBConfigRec *GLXFBConfig;
typedef unsigned long XID;

#define GL_RENDERER 0x1F01
#define GL_COMPUTE_SHADER 0x91B9
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#define GL_STATIC_DRAW 0x88E4
#define GL_BUFFER_UPDATE_BARRIER_BIT 0x00000200
#define GL_SHADER_STORAGE_BARRIER_BIT 0x00002000
#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
#define GL_SYNC_FLUSH_COMMANDS_BIT 0x00000001
#define GL_TIMEOUT_EXPIRED 0x911B
#define GL_WAIT_FAILED 0x911D
#define GL_TIME_ELAPSED 0x88BF
#define GL_QUERY_RESULT 0x8866
#define GLX_RENDER_TYPE 0x8011
#define GLX_RGBA_BIT 0x00000001
#define GLX_DRAWABLE_TYPE 0x8010
#define GLX_PBUFFER_BIT 0x00000004
#define GLX_PBUFFER_HEIGHT 0x8040
#define GLX_PBUFFER_WIDTH 0x8041
#define GLX_CONTEXT_MAJOR_VERSION_ARB 0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB 0x2092
#define GLX_CONTEXT_PROFILE_MASK_ARB 0x9126
#define GLX_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001

#define MAX_PROGRAMS 16
#define MAX_QUERIES 4096

struct jd_gl {
  void *x11, *lib;
  Display *dpy;
  GLXContext ctx;
  XID pb;
  struct { int dtype, tt; GLuint prog; GLint u_in, u_ntok, u_row0, u_nrows, u_ldy; } progs[MAX_PROGRAMS];
  int n_progs;
  GLuint queries[MAX_QUERIES];
  Display *(*XOpenDisplay)(const char *);
  int (*XCloseDisplay)(Display *);
  int (*XDefaultScreen)(Display *);
  void *(*glXGetProcAddressARB)(const GLubyte *);
  GLXFBConfig *(*glXChooseFBConfig)(Display *, int, const int *, int *);
  XID (*glXCreatePbuffer)(Display *, GLXFBConfig, const int *);
  void (*glXDestroyPbuffer)(Display *, XID);
  int (*glXMakeContextCurrent)(Display *, XID, XID, GLXContext);
  void (*glXDestroyContext)(Display *, GLXContext);
  GLXContext (*glXCreateContextAttribsARB)(Display *, GLXFBConfig, GLXContext, int, const int *);
  const GLubyte *(*GetString)(GLenum);
  GLenum (*GetError)(void);
  GLuint (*CreateShader)(GLenum);
  void (*ShaderSource)(GLuint, GLsizei, const GLchar *const *, const GLint *);
  void (*CompileShader)(GLuint);
  void (*GetShaderiv)(GLuint, GLenum, GLint *);
  void (*GetShaderInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *);
  void (*DeleteShader)(GLuint);
  GLuint (*CreateProgram)(void);
  void (*AttachShader)(GLuint, GLuint);
  void (*LinkProgram)(GLuint);
  void (*GetProgramiv)(GLuint, GLenum, GLint *);
  void (*GetProgramInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *);
  void (*UseProgram)(GLuint);
  void (*DeleteProgram)(GLuint);
  GLint (*GetUniformLocation)(GLuint, const GLchar *);
  void (*Uniform1i)(GLint, GLint);
  void (*GenBuffers)(GLsizei, GLuint *);
  void (*DeleteBuffers)(GLsizei, const GLuint *);
  void (*BindBuffer)(GLenum, GLuint);
  void (*BufferData)(GLenum, GLsizeiptr, const void *, GLenum);
  void (*BindBufferBase)(GLenum, GLuint, GLuint);
  void (*GetBufferSubData)(GLenum, GLintptr, GLsizeiptr, void *);
  void (*DispatchCompute)(GLuint, GLuint, GLuint);
  void (*MemoryBarrier)(GLbitfield);
  GLsync (*FenceSync)(GLenum, GLbitfield);
  GLenum (*ClientWaitSync)(GLsync, GLbitfield, GLuint64);
  void (*DeleteSync)(GLsync);
  void (*GenQueries)(GLsizei, GLuint *);
  void (*DeleteQueries)(GLsizei, const GLuint *);
  void (*BeginQuery)(GLenum, GLuint);
  void (*EndQuery)(GLenum);
  void (*GetQueryObjectui64v)(GLuint, GLenum, GLuint64 *);
};

static void *sym(jd_gl *g, const char *name) {
  void *p = g->glXGetProcAddressARB ? g->glXGetProcAddressARB((const GLubyte *)name) : NULL;
  return p ? p : dlsym(g->lib, name);
}

jd_gl *jd_gl_create(const char *libgl, char *err, int err_len) {
  jd_gl *g = calloc(1, sizeof(jd_gl));
  if (!g) return NULL;
#define FAIL(msg) do { snprintf(err, (size_t)err_len, "%s", msg); jd_gl_destroy(g); return NULL; } while (0)
  g->x11 = dlopen("libX11.so.6", RTLD_NOW | RTLD_GLOBAL);
  g->lib = dlopen(libgl ? libgl : "libGL.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!g->x11 || !g->lib) FAIL("cannot dlopen libX11.so.6 or libGL");
  *(void **)&g->XOpenDisplay = dlsym(g->x11, "XOpenDisplay");
  *(void **)&g->XCloseDisplay = dlsym(g->x11, "XCloseDisplay");
  *(void **)&g->XDefaultScreen = dlsym(g->x11, "XDefaultScreen");
  *(void **)&g->glXGetProcAddressARB = dlsym(g->lib, "glXGetProcAddressARB");
#define LOAD(field, name) *(void **)(&g->field) = sym(g, name)
  LOAD(glXChooseFBConfig, "glXChooseFBConfig"); LOAD(glXCreatePbuffer, "glXCreatePbuffer");
  LOAD(glXDestroyPbuffer, "glXDestroyPbuffer"); LOAD(glXMakeContextCurrent, "glXMakeContextCurrent");
  LOAD(glXDestroyContext, "glXDestroyContext"); LOAD(glXCreateContextAttribsARB, "glXCreateContextAttribsARB");
  if (!g->XOpenDisplay || !g->glXChooseFBConfig || !g->glXCreateContextAttribsARB) FAIL("GLX entry points missing");
  g->dpy = g->XOpenDisplay(NULL);
  if (!g->dpy) FAIL("XOpenDisplay failed");
  static const int fb_attr[] = {GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT, GLX_RENDER_TYPE, GLX_RGBA_BIT, 0};
  int n = 0;
  GLXFBConfig *fbc = g->glXChooseFBConfig(g->dpy, g->XDefaultScreen(g->dpy), fb_attr, &n);
  if (!fbc || !n) FAIL("no pbuffer FB config");
  static const int ctx_attr[] = {GLX_CONTEXT_MAJOR_VERSION_ARB, 4, GLX_CONTEXT_MINOR_VERSION_ARB, 3,
                                 GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB, 0};
  g->ctx = g->glXCreateContextAttribsARB(g->dpy, fbc[0], NULL, 1, ctx_attr);
  if (!g->ctx) FAIL("no OpenGL 4.3 core context");
  static const int pb_attr[] = {GLX_PBUFFER_WIDTH, 1, GLX_PBUFFER_HEIGHT, 1, 0};
  g->pb = g->glXCreatePbuffer(g->dpy, fbc[0], pb_attr);
  if (!g->glXMakeContextCurrent(g->dpy, g->pb, g->pb, g->ctx)) FAIL("glXMakeContextCurrent failed");
  LOAD(GetString, "glGetString"); LOAD(GetError, "glGetError");
  LOAD(CreateShader, "glCreateShader"); LOAD(ShaderSource, "glShaderSource");
  LOAD(CompileShader, "glCompileShader"); LOAD(GetShaderiv, "glGetShaderiv");
  LOAD(GetShaderInfoLog, "glGetShaderInfoLog"); LOAD(DeleteShader, "glDeleteShader");
  LOAD(CreateProgram, "glCreateProgram"); LOAD(AttachShader, "glAttachShader");
  LOAD(LinkProgram, "glLinkProgram"); LOAD(GetProgramiv, "glGetProgramiv");
  LOAD(GetProgramInfoLog, "glGetProgramInfoLog"); LOAD(UseProgram, "glUseProgram");
  LOAD(DeleteProgram, "glDeleteProgram"); LOAD(GetUniformLocation, "glGetUniformLocation");
  LOAD(Uniform1i, "glUniform1i"); LOAD(GenBuffers, "glGenBuffers"); LOAD(DeleteBuffers, "glDeleteBuffers");
  LOAD(BindBuffer, "glBindBuffer"); LOAD(BufferData, "glBufferData"); LOAD(BindBufferBase, "glBindBufferBase");
  LOAD(GetBufferSubData, "glGetBufferSubData"); LOAD(DispatchCompute, "glDispatchCompute");
  LOAD(MemoryBarrier, "glMemoryBarrier"); LOAD(FenceSync, "glFenceSync");
  LOAD(ClientWaitSync, "glClientWaitSync"); LOAD(DeleteSync, "glDeleteSync");
  LOAD(GenQueries, "glGenQueries"); LOAD(DeleteQueries, "glDeleteQueries");
  LOAD(BeginQuery, "glBeginQuery"); LOAD(EndQuery, "glEndQuery");
  LOAD(GetQueryObjectui64v, "glGetQueryObjectui64v");
#undef LOAD
  if (!g->DispatchCompute || !g->MemoryBarrier || !g->FenceSync) FAIL("no compute entry points");
  g->GenQueries(MAX_QUERIES, g->queries);
  return g;
#undef FAIL
}

void jd_gl_destroy(jd_gl *g) {
  if (!g) return;
  if (g->dpy && g->ctx) {
    for (int i = 0; i < g->n_progs; i++) g->DeleteProgram(g->progs[i].prog);
    if (g->DeleteQueries) g->DeleteQueries(MAX_QUERIES, g->queries);
    g->glXMakeContextCurrent(g->dpy, 0, 0, NULL);
    if (g->pb) g->glXDestroyPbuffer(g->dpy, g->pb);
    g->glXDestroyContext(g->dpy, g->ctx);
  }
  if (g->dpy) g->XCloseDisplay(g->dpy);
  free(g);
}

const char *jd_gl_renderer(jd_gl *g) { return (const char *)g->GetString(GL_RENDERER); }

/* ---- buffers ------------------------------------------------------------------------------- */

unsigned int jd_gl_buffer(jd_gl *g, const void *data, long bytes) {
  GLuint b = 0;
  g->GenBuffers(1, &b);
  g->BindBuffer(GL_SHADER_STORAGE_BUFFER, b);
  g->BufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)bytes, data, GL_STATIC_DRAW);
  return b;
}

void jd_gl_free_buffer(jd_gl *g, unsigned int buf) { g->DeleteBuffers(1, &buf); }

int jd_gl_read(jd_gl *g, unsigned int buf, void *dst, long bytes) {
  g->MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT); /* shader writes -> buffer reads */
  g->BindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
  g->GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)bytes, dst);
  return g->GetError() == 0 ? 0 : -1;
}

int jd_gl_upload_matrix(jd_gl *g, const jd_tensor *w, jd_gl_matrix *out) {
  memset(out, 0, sizeof(*out));
  out->dtype = w->dtype;
  out->rows = (int)w->dims[0];
  out->cols = (int)w->dims[1];
  if (w->dtype == JD_F32) {
    out->buf_w = jd_gl_buffer(g, w->data, (long)w->nbytes);
    return 0;
  }
  if (w->dtype != JD_Q8 && w->dtype != JD_Q4) return -1;
  int nb = out->cols / 32, per = w->dtype == JD_Q8 ? 8 : 4, bsz = w->dtype == JD_Q8 ? 34 : 18;
  long nblocks = (long)out->rows * nb;
  if (nblocks % 2) return -1; /* scales pack in pairs */
  uint32_t *q = malloc((size_t)nblocks * per * 4), *s = calloc((size_t)nblocks / 2, 4);
  if (!q || !s) { free(q); free(s); return -1; }
  const unsigned char *src = w->data;
  for (long b = 0; b < nblocks; b++, src += bsz) {
    uint32_t half = (uint32_t)src[0] | ((uint32_t)src[1] << 8);
    s[b / 2] |= half << (16 * (b & 1));
    memcpy(q + b * per, src + 2, (size_t)per * 4);
  }
  out->buf_w = jd_gl_buffer(g, q, nblocks * per * 4);
  out->buf_s = jd_gl_buffer(g, s, nblocks / 2 * 4);
  free(q);
  free(s);
  return g->GetError() == 0 ? 0 : -1;
}

void jd_gl_free_matrix(jd_gl *g, jd_gl_matrix *m) {
  if (m->buf_w) jd_gl_free_buffer(g, m->buf_w);
  if (m->buf_s) jd_gl_free_buffer(g, m->buf_s);
  memset(m, 0, sizeof(*m));
}

/* ---- the matmul shader ---------------------------------------------------------------------
 * One invocation per output row; each holds TT tokens' accumulators. A workgroup of 64 rows
 * walks the input in blocks of 32: it stages TT x 32 activations in shared memory, then every
 * invocation dequantises its row's block (fp16 scale, 4- or 8-bit values) and accumulates in
 * fp32. Plain shared memory and barrier(); no subgroup operations. */

static const char *SHADER_HEAD =
    "#version 430\n"
    "layout(local_size_x = 64) in;\n"
    "layout(std430, binding = 0) readonly buffer Wq { uint wq[]; };\n"
    "layout(std430, binding = 1) readonly buffer Ws { uint ws[]; };\n"
    "layout(std430, binding = 2) readonly buffer X { float x[]; };\n"
    "layout(std430, binding = 3) writeonly buffer Y { float y[]; };\n"
    "uniform int in_dim, n_tok, row0, n_rows, ldy;\n"
    "shared float xs[TT * 32];\n"
    "void main() {\n"
    "  int lr = int(gl_LocalInvocationID.x);\n"
    "  int row = row0 + int(gl_WorkGroupID.x) * 64 + lr;\n"
    "  int t0 = int(gl_WorkGroupID.y) * TT;\n"
    "  int nb = in_dim / 32;\n"
    "  bool live = row < row0 + n_rows;\n"
    "  float acc[TT];\n"
    "  for (int t = 0; t < TT; t++) acc[t] = 0.0;\n"
    "  for (int b = 0; b < nb; b++) {\n"
    "    for (int k = lr; k < TT * 32; k += 64) {\n"
    "      int tok = t0 + k / 32;\n"
    "      xs[k] = tok < n_tok ? x[tok * in_dim + b * 32 + (k % 32)] : 0.0;\n"
    "    }\n"
    "    barrier();\n"
    "    if (live) {\n";

static const char *SHADER_Q4 =
    "      int blk = row * nb + b;\n"
    "      float s = unpackHalf2x16(ws[blk / 2])[blk & 1];\n"
    "      for (int j = 0; j < 4; j++) {\n"
    "        uint w = wq[blk * 4 + j];\n"
    "        for (int k = 0; k < 8; k++) {\n"
    "          float wv = (float((w >> uint(4 * k)) & 15u) - 8.0) * s;\n"
    "          for (int t = 0; t < TT; t++) acc[t] += wv * xs[t * 32 + j * 8 + k];\n"
    "        }\n"
    "      }\n";

static const char *SHADER_Q8 =
    "      int blk = row * nb + b;\n"
    "      float s = unpackHalf2x16(ws[blk / 2])[blk & 1];\n"
    "      for (int j = 0; j < 8; j++) {\n"
    "        int w = int(wq[blk * 8 + j]);\n"
    "        for (int k = 0; k < 4; k++) {\n"
    "          float wv = float(bitfieldExtract(w, 8 * k, 8)) * s;\n"
    "          for (int t = 0; t < TT; t++) acc[t] += wv * xs[t * 32 + j * 4 + k];\n"
    "        }\n"
    "      }\n";

static const char *SHADER_F32 =
    "      for (int i = 0; i < 32; i++) {\n"
    "        float wv = uintBitsToFloat(wq[row * in_dim + b * 32 + i]);\n"
    "        for (int t = 0; t < TT; t++) acc[t] += wv * xs[t * 32 + i];\n"
    "      }\n";

static const char *SHADER_TAIL =
    "    }\n"
    "    barrier();\n"
    "  }\n"
    "  if (live)\n"
    "    for (int t = 0; t < TT; t++)\n"
    "      if (t0 + t < n_tok) y[(t0 + t) * ldy + row] = acc[t];\n"
    "}\n";

static int program_for(jd_gl *g, int dtype, int tt) {
  for (int i = 0; i < g->n_progs; i++)
    if (g->progs[i].dtype == dtype && g->progs[i].tt == tt) return i;
  if (g->n_progs == MAX_PROGRAMS) return -1;
  char def[64];
  snprintf(def, sizeof(def), "#define TT %d\n", tt);
  const char *body = dtype == JD_Q4 ? SHADER_Q4 : dtype == JD_Q8 ? SHADER_Q8 : SHADER_F32;
  /* the #define must follow #version: splice it after the first line */
  size_t vl = strchr(SHADER_HEAD, '\n') - SHADER_HEAD + 1;
  size_t len = strlen(SHADER_HEAD) + strlen(def) + strlen(body) + strlen(SHADER_TAIL) + 1;
  char *src = malloc(len);
  if (!src) return -1;
  memcpy(src, SHADER_HEAD, vl);
  src[vl] = 0;
  strcat(src, def);
  strcat(src, SHADER_HEAD + vl);
  strcat(src, body);
  strcat(src, SHADER_TAIL);
  GLuint sh = g->CreateShader(GL_COMPUTE_SHADER);
  const char *srcs[1] = {src};
  g->ShaderSource(sh, 1, srcs, NULL);
  g->CompileShader(sh);
  GLint ok = 0;
  g->GetShaderiv(sh, GL_COMPILE_STATUS, &ok);
  char log[2048] = {0};
  if (!ok) {
    g->GetShaderInfoLog(sh, sizeof(log), NULL, log);
    fprintf(stderr, "[jd_gl] compile failed:\n%s\n", log);
    free(src);
    return -1;
  }
  GLuint p = g->CreateProgram();
  g->AttachShader(p, sh);
  g->LinkProgram(p);
  g->DeleteShader(sh);
  free(src);
  g->GetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    g->GetProgramInfoLog(p, sizeof(log), NULL, log);
    fprintf(stderr, "[jd_gl] link failed:\n%s\n", log);
    return -1;
  }
  int i = g->n_progs++;
  g->progs[i].dtype = dtype;
  g->progs[i].tt = tt;
  g->progs[i].prog = p;
  g->progs[i].u_in = g->GetUniformLocation(p, "in_dim");
  g->progs[i].u_ntok = g->GetUniformLocation(p, "n_tok");
  g->progs[i].u_row0 = g->GetUniformLocation(p, "row0");
  g->progs[i].u_nrows = g->GetUniformLocation(p, "n_rows");
  g->progs[i].u_ldy = g->GetUniformLocation(p, "ldy");
  return i;
}

static int wait_gpu(jd_gl *g) {
  GLsync f = g->FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  /* block in the driver, with a real timeout, rather than poll */
  GLenum r = g->ClientWaitSync(f, GL_SYNC_FLUSH_COMMANDS_BIT, 5000000000ull);
  g->DeleteSync(f);
  return r == GL_TIMEOUT_EXPIRED || r == GL_WAIT_FAILED ? -1 : 0;
}

int jd_gl_matmul(jd_gl *g, const jd_gl_matrix *w, unsigned int x, int n, unsigned int y, int ldy,
                 jd_gl_tiling tl, double *gpu_ms, int *dispatches) {
  if (tl.tokens_per_thread <= 0) tl.tokens_per_thread = 8;
  if (tl.rows_per_dispatch <= 0) tl.rows_per_dispatch = w->rows;
  if (tl.dispatches_per_fence <= 0) tl.dispatches_per_fence = 1;
  tl.rows_per_dispatch = (tl.rows_per_dispatch + 63) / 64 * 64;
  int pi = program_for(g, w->dtype, tl.tokens_per_thread);
  if (pi < 0) return -1;
  g->UseProgram(g->progs[pi].prog);
  g->Uniform1i(g->progs[pi].u_in, w->cols);
  g->Uniform1i(g->progs[pi].u_ntok, n);
  g->Uniform1i(g->progs[pi].u_ldy, ldy);
  g->BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, w->buf_w);
  g->BindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, w->buf_s ? w->buf_s : w->buf_w);
  g->BindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, x);
  g->BindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, y);
  GLuint gy = (GLuint)((n + tl.tokens_per_thread - 1) / tl.tokens_per_thread);
  int nd = 0, pending = 0, first_q = 0;
  double total = 0;
  for (int r0 = 0; r0 < w->rows; r0 += tl.rows_per_dispatch) {
    int nr = w->rows - r0 < tl.rows_per_dispatch ? w->rows - r0 : tl.rows_per_dispatch;
    g->Uniform1i(g->progs[pi].u_row0, r0);
    g->Uniform1i(g->progs[pi].u_nrows, nr);
    GLuint q = g->queries[nd % MAX_QUERIES];
    g->BeginQuery(GL_TIME_ELAPSED, q);
    g->DispatchCompute((GLuint)((nr + 63) / 64), gy, 1);
    g->EndQuery(GL_TIME_ELAPSED);
    nd++;
    if (++pending == tl.dispatches_per_fence || r0 + tl.rows_per_dispatch >= w->rows) {
      if (wait_gpu(g) != 0) return -1;
      for (int k = first_q; k < nd; k++) {
        GLuint64 ns = 0;
        g->GetQueryObjectui64v(g->queries[k % MAX_QUERIES], GL_QUERY_RESULT, &ns);
        total += ns / 1e6;
      }
      first_q = nd;
      pending = 0;
    }
  }
  g->MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); /* for a later shader reading y */
  if (gpu_ms) *gpu_ms = total;
  if (dispatches) *dispatches = nd;
  return g->GetError() == 0 ? 0 : -1;
}
