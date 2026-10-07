/* jibo-gl-probe: does our own desktop-GL compute work on this GPU, and how long does a dispatch take?
 *
 *   DISPLAY=:0 jibo-gl-probe [--libgl PATH] [--busy-iters N] [--reps N]
 *
 * The first GPU deliverable of docs/jibo-port.md (Phase 5). A separate diagnostic: it creates its
 * own OpenGL 4.3 core context on a 1x1 pbuffer and touches nothing else. Run it under the same
 * display access and library shim that the game host's run.sh arranges; DISPLAY alone may not be
 * enough. libGL and libX11 are opened with dlopen, so the binary links against libc and libdl only
 * and builds without GL headers.
 *
 * It prints one JSON object per line: the GL strings, the compute limits, the correctness check
 * (values[i] = 3i + 7 over 256 uints, 4 workgroups of 64, every value compared), and timings: host
 * latency of dispatch + barrier + fence + readback, GPU time from GL_TIME_ELAPSED queries, for an
 * empty dispatch and a busy one. Exit status 0 only if the check passes.
 */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- the GL and GLX we use, declared here ------------------------------------------------- */
typedef unsigned int GLenum, GLuint, GLbitfield;
typedef int GLint, GLsizei;
typedef char GLchar;
typedef unsigned char GLubyte;
typedef intptr_t GLintptr;
typedef intptr_t GLsizeiptr;
typedef uint64_t GLuint64;
typedef struct __GLsync *GLsync;
typedef struct _XDisplay Display;
typedef struct __GLXcontextRec *GLXContext;
typedef struct __GLXFBConfigRec *GLXFBConfig;
typedef unsigned long XID;

#define GL_NO_ERROR 0
#define GL_VENDOR 0x1F00
#define GL_RENDERER 0x1F01
#define GL_VERSION 0x1F02
#define GL_SHADING_LANGUAGE_VERSION 0x8B8C
#define GL_NUM_EXTENSIONS 0x821D
#define GL_EXTENSIONS 0x1F03
#define GL_COMPUTE_SHADER 0x91B9
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#define GL_DYNAMIC_READ 0x88E9
#define GL_MAX_COMPUTE_WORK_GROUP_COUNT 0x91BE
#define GL_MAX_COMPUTE_WORK_GROUP_SIZE 0x91BF
#define GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS 0x90EB
#define GL_MAX_COMPUTE_SHARED_MEMORY_SIZE 0x8262
#define GL_MAX_SHADER_STORAGE_BLOCK_SIZE 0x90DE
#define GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS 0x90DD
#define GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS 0x90DB
#define GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT 0x90DF
#define GL_BUFFER_UPDATE_BARRIER_BIT 0x00000200
#define GL_SHADER_STORAGE_BARRIER_BIT 0x00002000
#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
#define GL_SYNC_FLUSH_COMMANDS_BIT 0x00000001
#define GL_ALREADY_SIGNALED 0x911A
#define GL_TIMEOUT_EXPIRED 0x911B
#define GL_CONDITION_SATISFIED 0x911C
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

static struct {
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
  const GLubyte *(*GetStringi)(GLenum, GLuint);
  void (*GetIntegerv)(GLenum, GLint *);
  void (*GetIntegeri_v)(GLenum, GLuint, GLint *);
  GLenum (*GetError)(void);
  GLuint (*CreateShader)(GLenum);
  void (*ShaderSource)(GLuint, GLsizei, const GLchar *const *, const GLint *);
  void (*CompileShader)(GLuint);
  void (*GetShaderiv)(GLuint, GLenum, GLint *);
  void (*GetShaderInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *);
  GLuint (*CreateProgram)(void);
  void (*AttachShader)(GLuint, GLuint);
  void (*LinkProgram)(GLuint);
  void (*GetProgramiv)(GLuint, GLenum, GLint *);
  void (*GetProgramInfoLog)(GLuint, GLsizei, GLsizei *, GLchar *);
  void (*UseProgram)(GLuint);
  GLint (*GetUniformLocation)(GLuint, const GLchar *);
  void (*Uniform1ui)(GLint, GLuint);
  void (*GenBuffers)(GLsizei, GLuint *);
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
  void (*BeginQuery)(GLenum, GLuint);
  void (*EndQuery)(GLenum);
  void (*GetQueryObjectui64v)(GLuint, GLenum, GLuint64 *);
  void (*Finish)(void);
} gl;

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void json_str(const char *k, const char *v, int last) {
  printf("\"%s\":\"", k);
  for (const char *p = v ? v : ""; *p; p++) {
    if (*p == '"' || *p == '\\') printf("\\%c", *p);
    else if ((unsigned char)*p < 0x20) printf("\\u%04x", *p);
    else putchar(*p);
  }
  printf(last ? "\"" : "\",");
}

static int fail(const char *stage, const char *msg) {
  printf("{\"probe\":\"error\",");
  json_str("stage", stage, 0);
  json_str("message", msg, 1);
  printf("}\n");
  return 1;
}

static void *sym(void *lib, const char *name) {
  void *p = gl.glXGetProcAddressARB ? gl.glXGetProcAddressARB((const GLubyte *)name) : NULL;
  return p ? p : dlsym(lib, name);
}

static GLuint compile(const char *src, const char *name) {
  GLuint s = gl.CreateShader(GL_COMPUTE_SHADER), p;
  GLint ok = 0;
  char log[4096] = {0};
  gl.ShaderSource(s, 1, &src, NULL);
  gl.CompileShader(s);
  gl.GetShaderiv(s, GL_COMPILE_STATUS, &ok);
  gl.GetShaderInfoLog(s, sizeof(log), NULL, log);
  printf("{\"probe\":\"compile\",");
  json_str("shader", name, 0);
  printf("\"ok\":%s,", ok ? "true" : "false");
  json_str("log", log, 1);
  printf("}\n");
  if (!ok) return 0;
  p = gl.CreateProgram();
  gl.AttachShader(p, s);
  gl.LinkProgram(p);
  gl.GetProgramiv(p, GL_LINK_STATUS, &ok);
  log[0] = 0;
  gl.GetProgramInfoLog(p, sizeof(log), NULL, log);
  printf("{\"probe\":\"link\",");
  json_str("shader", name, 0);
  printf("\"ok\":%s,", ok ? "true" : "false");
  json_str("log", log, 1);
  printf("}\n");
  return ok ? p : 0;
}

/* Wait for the GPU without spinning: block in the driver with a real timeout. */
static GLenum wait_fence(double *ms) {
  double t = now_ms();
  GLsync f = gl.FenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  GLenum r = gl.ClientWaitSync(f, GL_SYNC_FLUSH_COMMANDS_BIT, 2000000000ull); /* 2 s */
  gl.DeleteSync(f);
  *ms = now_ms() - t;
  return r;
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

static void timed(const char *name, GLuint prog, GLint iters_loc, unsigned iters, int reps, GLuint query) {
  double *host = malloc(sizeof(double) * (size_t)reps), *gpu = malloc(sizeof(double) * (size_t)reps);
  if (!host || !gpu) return;
  gl.UseProgram(prog);
  if (iters_loc >= 0) gl.Uniform1ui(iters_loc, iters);
  for (int r = 0; r < reps; r++) {
    double t0 = now_ms(), w;
    gl.BeginQuery(GL_TIME_ELAPSED, query);
    gl.DispatchCompute(4, 1, 1);
    gl.EndQuery(GL_TIME_ELAPSED);
    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    wait_fence(&w);
    host[r] = now_ms() - t0;
    GLuint64 ns = 0;
    gl.GetQueryObjectui64v(query, GL_QUERY_RESULT, &ns);
    gpu[r] = ns / 1e6;
  }
  qsort(host, (size_t)reps, sizeof(double), cmp_double);
  qsort(gpu, (size_t)reps, sizeof(double), cmp_double);
  printf("{\"probe\":\"timing\",\"kernel\":\"%s\",\"busy_iters\":%u,\"reps\":%d,"
         "\"host_ms_p50\":%.4f,\"host_ms_p95\":%.4f,\"host_ms_max\":%.4f,"
         "\"gpu_ms_p50\":%.4f,\"gpu_ms_p95\":%.4f,\"gpu_ms_max\":%.4f}\n",
         name, iters, reps, host[reps / 2], host[(reps * 95) / 100], host[reps - 1], gpu[reps / 2],
         gpu[(reps * 95) / 100], gpu[reps - 1]);
  free(host);
  free(gpu);
}

static const char *SHADER_CHECK =
    "#version 430\n"
    "layout(local_size_x = 64) in;\n"
    "layout(std430, binding = 0) buffer Result { uint values[]; };\n"
    "void main() {\n"
    "    uint i = gl_GlobalInvocationID.x;\n"
    "    if (i < 256u) values[i] = 3u * i + 7u;\n"
    "}\n";

/* A dispatch of adjustable length: each invocation runs a dependent integer loop. */
static const char *SHADER_BUSY =
    "#version 430\n"
    "layout(local_size_x = 64) in;\n"
    "layout(std430, binding = 0) buffer Result { uint values[]; };\n"
    "uniform uint iters;\n"
    "void main() {\n"
    "    uint i = gl_GlobalInvocationID.x, x = i;\n"
    "    for (uint k = 0u; k < iters; k++) x = x * 1664525u + 1013904223u;\n"
    "    if (i < 256u) values[i] = x;\n"
    "}\n";

int main(int argc, char **argv) {
  const char *libgl_path = getenv("JIBO_LIBGL") ? getenv("JIBO_LIBGL") : "libGL.so.1";
  unsigned busy = 20000;
  int reps = 50;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--libgl") && i + 1 < argc) libgl_path = argv[++i];
    else if (!strcmp(argv[i], "--busy-iters") && i + 1 < argc) busy = (unsigned)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else { fprintf(stderr, "usage: jibo-gl-probe [--libgl PATH] [--busy-iters N] [--reps N]\n"); return 2; }
  }
  if (reps < 1) reps = 1;
  void *x11 = dlopen("libX11.so.6", RTLD_NOW | RTLD_GLOBAL), *lgl = dlopen(libgl_path, RTLD_NOW | RTLD_GLOBAL);
  if (!x11 || !lgl) return fail("dlopen", dlerror());
  gl.XOpenDisplay = (Display * (*)(const char *)) dlsym(x11, "XOpenDisplay");
  gl.XCloseDisplay = (int (*)(Display *))dlsym(x11, "XCloseDisplay");
  gl.XDefaultScreen = (int (*)(Display *))dlsym(x11, "XDefaultScreen");
  gl.glXGetProcAddressARB = (void *(*)(const GLubyte *))dlsym(lgl, "glXGetProcAddressARB");
#define LOAD(field, name) *(void **)(&gl.field) = sym(lgl, name)
  LOAD(glXChooseFBConfig, "glXChooseFBConfig");
  LOAD(glXCreatePbuffer, "glXCreatePbuffer");
  LOAD(glXDestroyPbuffer, "glXDestroyPbuffer");
  LOAD(glXMakeContextCurrent, "glXMakeContextCurrent");
  LOAD(glXDestroyContext, "glXDestroyContext");
  LOAD(glXCreateContextAttribsARB, "glXCreateContextAttribsARB");
  if (!gl.XOpenDisplay || !gl.glXChooseFBConfig || !gl.glXCreateContextAttribsARB)
    return fail("dlsym", "GLX entry points missing");

  Display *dpy = gl.XOpenDisplay(NULL);
  if (!dpy) return fail("display", "XOpenDisplay failed; check DISPLAY and the launcher's display access");
  static const int fb_attr[] = {GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT, GLX_RENDER_TYPE, GLX_RGBA_BIT, 0};
  int n = 0;
  GLXFBConfig *fbc = gl.glXChooseFBConfig(dpy, gl.XDefaultScreen(dpy), fb_attr, &n);
  if (!fbc || n == 0) return fail("fbconfig", "no pbuffer-capable FB config");
  static const int ctx_attr[] = {GLX_CONTEXT_MAJOR_VERSION_ARB, 4, GLX_CONTEXT_MINOR_VERSION_ARB, 3,
                                 GLX_CONTEXT_PROFILE_MASK_ARB, GLX_CONTEXT_CORE_PROFILE_BIT_ARB, 0};
  GLXContext ctx = gl.glXCreateContextAttribsARB(dpy, fbc[0], NULL, 1, ctx_attr);
  if (!ctx) return fail("context", "no OpenGL 4.3 core context");
  static const int pb_attr[] = {GLX_PBUFFER_WIDTH, 1, GLX_PBUFFER_HEIGHT, 1, 0};
  XID pb = gl.glXCreatePbuffer(dpy, fbc[0], pb_attr);
  if (!gl.glXMakeContextCurrent(dpy, pb, pb, ctx)) return fail("context", "glXMakeContextCurrent failed");

  LOAD(GetString, "glGetString"); LOAD(GetStringi, "glGetStringi"); LOAD(GetIntegerv, "glGetIntegerv");
  LOAD(GetIntegeri_v, "glGetIntegeri_v"); LOAD(GetError, "glGetError");
  LOAD(CreateShader, "glCreateShader"); LOAD(ShaderSource, "glShaderSource");
  LOAD(CompileShader, "glCompileShader"); LOAD(GetShaderiv, "glGetShaderiv");
  LOAD(GetShaderInfoLog, "glGetShaderInfoLog"); LOAD(CreateProgram, "glCreateProgram");
  LOAD(AttachShader, "glAttachShader"); LOAD(LinkProgram, "glLinkProgram");
  LOAD(GetProgramiv, "glGetProgramiv"); LOAD(GetProgramInfoLog, "glGetProgramInfoLog");
  LOAD(UseProgram, "glUseProgram"); LOAD(GetUniformLocation, "glGetUniformLocation");
  LOAD(Uniform1ui, "glUniform1ui"); LOAD(GenBuffers, "glGenBuffers"); LOAD(BindBuffer, "glBindBuffer");
  LOAD(BufferData, "glBufferData"); LOAD(BindBufferBase, "glBindBufferBase");
  LOAD(GetBufferSubData, "glGetBufferSubData"); LOAD(DispatchCompute, "glDispatchCompute");
  LOAD(MemoryBarrier, "glMemoryBarrier"); LOAD(FenceSync, "glFenceSync");
  LOAD(ClientWaitSync, "glClientWaitSync"); LOAD(DeleteSync, "glDeleteSync");
  LOAD(GenQueries, "glGenQueries"); LOAD(BeginQuery, "glBeginQuery"); LOAD(EndQuery, "glEndQuery");
  LOAD(GetQueryObjectui64v, "glGetQueryObjectui64v"); LOAD(Finish, "glFinish");
#undef LOAD
  if (!gl.DispatchCompute || !gl.MemoryBarrier || !gl.FenceSync) return fail("entry points", "no compute");

  printf("{\"probe\":\"strings\",");
  json_str("vendor", (const char *)gl.GetString(GL_VENDOR), 0);
  json_str("renderer", (const char *)gl.GetString(GL_RENDERER), 0);
  json_str("version", (const char *)gl.GetString(GL_VERSION), 0);
  json_str("glsl", (const char *)gl.GetString(GL_SHADING_LANGUAGE_VERSION), 1);
  printf("}\n");
  GLint next = 0, has_compute = 0, has_ssbo = 0;
  gl.GetIntegerv(GL_NUM_EXTENSIONS, &next);
  for (GLint i = 0; i < next; i++) {
    const char *e = (const char *)gl.GetStringi(GL_EXTENSIONS, (GLuint)i);
    if (e && !strcmp(e, "GL_ARB_compute_shader")) has_compute = 1;
    if (e && !strcmp(e, "GL_ARB_shader_storage_buffer_object")) has_ssbo = 1;
  }
  GLint cnt[3], size[3], inv = 0, shared = 0, ssbo_size = 0, ssbo_bind = 0, ssbo_blocks = 0, align = 0;
  for (GLuint i = 0; i < 3; i++) {
    gl.GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, i, &cnt[i]);
    gl.GetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &size[i]);
  }
  gl.GetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &inv);
  gl.GetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &shared);
  gl.GetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &ssbo_size);
  gl.GetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &ssbo_bind);
  gl.GetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &ssbo_blocks);
  gl.GetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &align);
  printf("{\"probe\":\"limits\",\"extensions\":%d,\"ARB_compute_shader\":%s,\"ARB_shader_storage_buffer_object\":%s,"
         "\"max_work_group_count\":[%d,%d,%d],\"max_work_group_size\":[%d,%d,%d],"
         "\"max_work_group_invocations\":%d,\"max_shared_memory_bytes\":%d,\"max_ssbo_block_bytes\":%d,"
         "\"max_ssbo_bindings\":%d,\"max_compute_ssbo_blocks\":%d,\"ssbo_offset_alignment\":%d,\"gl_error\":%u}\n",
         next, has_compute ? "true" : "false", has_ssbo ? "true" : "false", cnt[0], cnt[1], cnt[2], size[0],
         size[1], size[2], inv, shared, ssbo_size, ssbo_bind, ssbo_blocks, align, gl.GetError());

  GLuint check = compile(SHADER_CHECK, "check"), busy_prog = compile(SHADER_BUSY, "busy");
  if (!check || !busy_prog) return fail("compile", "see the compile and link lines");
  GLuint buf, query;
  uint32_t init[256], out[256];
  memset(init, 0xFF, sizeof(init));
  gl.GenBuffers(1, &buf);
  gl.BindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
  gl.BufferData(GL_SHADER_STORAGE_BUFFER, sizeof(init), init, GL_DYNAMIC_READ);
  gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buf);
  gl.GenQueries(1, &query);

  double t0 = now_ms(), w;
  gl.UseProgram(check);
  gl.DispatchCompute(4, 1, 1);
  /* the readback is a buffer update: make the shader's writes visible to it */
  gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
  GLenum wr = wait_fence(&w);
  gl.GetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(out), out);
  double total = now_ms() - t0;
  GLenum err = gl.GetError();
  int bad = 0, first_bad = -1;
  for (int i = 0; i < 256; i++)
    if (out[i] != 3u * (uint32_t)i + 7u) { bad++; if (first_bad < 0) first_bad = i; }
  printf("{\"probe\":\"check\",\"ok\":%s,\"mismatches\":%d,\"first_mismatch\":%d,\"fence\":\"%s\","
         "\"fence_wait_ms\":%.4f,\"host_ms\":%.4f,\"gl_error\":%u}\n",
         bad == 0 && err == GL_NO_ERROR ? "true" : "false", bad, first_bad,
         wr == GL_ALREADY_SIGNALED ? "already_signaled" : wr == GL_CONDITION_SATISFIED ? "satisfied"
         : wr == GL_TIMEOUT_EXPIRED ? "timeout" : "failed", w, total, err);

  timed("empty", busy_prog, gl.GetUniformLocation(busy_prog, "iters"), 0, reps, query);
  timed("busy", busy_prog, gl.GetUniformLocation(busy_prog, "iters"), busy, reps, query);

  gl.glXMakeContextCurrent(dpy, 0, 0, NULL);
  gl.glXDestroyPbuffer(dpy, pb);
  gl.glXDestroyContext(dpy, ctx);
  gl.XCloseDisplay(dpy);
  return bad == 0 && err == GL_NO_ERROR ? 0 : 1;
}
