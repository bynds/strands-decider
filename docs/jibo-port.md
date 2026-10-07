# Porting the decider to Jibo: design and plan

This document designs a native port of the Strands Decider inference engine to the original
Jibo robot, and plans the work in phases with exit criteria. It is a proposal. None of the
runtime, tools or files it names exist yet, and no number in it was measured on a Jibo.
Paths are relative to the repository root unless they are links.

Source of the hardware facts: the owner's handoff of 7 October 2026 ("native inference and GPU
acceleration on original Jibo"). Its observations of the robot are the authority for this
design and are not re-derived here. The handoff named Needle 3 (`needle-rs`) as its default
model; this design ports this repository's engine instead. The handoff's hardware facts,
resource envelope, GPU bring-up order and deployment boundaries carry over unchanged. Its
Needle-specific build and kernel plan (handoff sections 3 and 6) is replaced by the runtime
below, and its ncnn alternative (section 7) does not apply.

- [Summary](#summary)
- [What runs today, and what has to move](#what-runs-today-and-what-has-to-move)
- [The target](#the-target)
- [Budget: what fits](#budget-what-fits)
- [Design](#design)
- [Numerical parity](#numerical-parity)
- [Plan](#plan)
- [Coexistence and safety rules](#coexistence-and-safety-rules)
- [Risks](#risks)
- [Decisions](#decisions)
- [Alternatives considered](#alternatives-considered)

## Summary

- **The released 2B checkpoint does not fit.** Qwen3.5-2B's decoder layers hold 1.37 billion
  parameters, all of which every forward reads. At 4 bits that is about 770 MB, more than the
  600 MB of headroom the robot has with the eye and vision running. The port therefore targets
  a decider trained on **Qwen3.5-0.8B-Base** (0.50 billion layer parameters, about 280 MB at
  4 bits). Training it is a preregistered run of the existing recipe with `base_model`
  changed. Its accuracy is unknown until measured.
- **A small native runtime in C99**, built with the existing `jibo-armcc` toolchain (Linaro GCC
  4.8.4). It contains the tokenizer, the prompt rendering of `prompting.py`, the Qwen3.5 forward,
  the pointer head and the answer formulas of `schema.py`. It depends only on libc, libm and
  libpthread. libGL is loaded with `dlopen`, and only by the GPU backend. With no C++ standard
  library to link, the robot's `GLIBCXX_3.4.20` ceiling cannot be exceeded.
- **One forward driver, two backends.** A CPU backend (scalar and NEON) is the reference and
  the fallback. A desktop-OpenGL 4.3 compute backend comes second, through the same op
  interface, so both execute the same graph. A CUDA Driver API backend could be a third,
  later.
- **Parity with the Python engine is the spine of the work.** An exporter in this repository
  writes the weights, tokenizer tables and golden tensors from the torch engine. The C runtime
  must reproduce them on x86 in fp32 before quantisation, NEON or the GPU enter, and then
  each of those changes is measured on its own.
- **Expect seconds per decision, not milliseconds.** At 128 prompt tokens the 0.8B forward is
  about 128 GFLOP. Arithmetic on the vendor peaks gives roughly 13 to 26 s on the one CPU core
  the robot can spare and roughly 2 to 6 s on the GPU. These are estimates for planning;
  Phase 6 measures the real figures. Several questions about one state share its encoding, as
  they do on the desktop.
- **The immediate milestone is the handoff's, adapted:** an ABI-clean ARMv7 CPU build of the
  runtime that matches the reference, a correct desktop-GL compute dispatch of our own, and a
  CPU-against-GPU benchmark of one real operation of this model (an MLP projection).

## What runs today, and what has to move

One request is a state and N typed questions (`noul`, `choice`, `score`). `SystemOneEngine` in
`src/strands_decider/infer.py` serves it in five steps:

1. **Render** (`prompting.py`): `<state>…</state>` then, per question, a `<question>` block whose
   options are numbered lines, ending in `<answer>`. Each option's character span is recorded.
2. **Fit and tokenise** (`_fit`): the question claims the window first (up to 75 % of
   `max_length`), the state takes the rest, and token offsets map each option span to the last
   token of its line.
3. **Torso forward**: Qwen3.5's text decoder, `Qwen3_5ForCausalLM.model`, with the LoRA adapter.
   With more than one question, the state is encoded once and its cache (keys and values for
   the 6 attention layers, convolution and recurrent states for the 18 Gated DeltaNet layers) is
   forked across the question suffixes (`_slot_probs_shared_prefix`). A single question forwards
   the whole prompt.
4. **Pointer head** (`modeling.PointerHead`, fp32): LayerNorm, then `q(h_answer) · k(h_option_k)
   / sqrt(256)` for each option, divided by the fitted temperature for the question's kind, then
   a masked softmax.
5. **Answers** (`_to_answer`, `schema.py`): P(true) for a noul, the argmax and the normalised
   max-probability confidence for a choice, and the expected level and the ordinal confidence
   for a score.

Only step 3 is expensive. Steps 1, 2, 4 and 5 are small, but their exact behaviour is part of
the model: a prompt rendered or tokenised differently is a different input, and the readout
fails when an option's last token is misplaced. The port reproduces all five steps. The
vision tower (`vision.py`) is out of scope, because Jibo's camera pipeline is a separate
service and the tower alone has hundreds of millions of parameters.

The LM head is never used: it is tied to the input embeddings, and the decider reads hidden
states instead. So the 248,320-row embedding table is used only for row lookups, one row per
input token. It needs no compute and almost no resident memory if it is memory-mapped from
flash.

## The target

The owner's verified facts, from the handoff, that shape the design:

| Item | State |
|---|---|
| CPU | 32-bit ARMv7 (Tegra K1, Cortex-A15 class), NEON, hard float, 4 active cores at 1.938 GHz. `arm-linux-gnueabihf`. |
| ABI ceiling | glibc 2.21; libstdc++ up to `GLIBCXX_3.4.20`. No headers on the robot. |
| Toolchain | `jibo-armcc` image, Linaro GCC 4.8.4, with an existing ABI check. Sysroot from `work/fs/p2`. |
| GPU | GK20A (Kepler, `sm_32`), one SMX of 192 lanes, unified memory with the CPU. Desktop OpenGL 4.4 with `GL_ARB_compute_shader` and SSBOs. OpenGL ES is 3.0 only, so no ES compute. |
| CUDA | Driver `libcuda.so.1.1` present; no toolkit and no shared runtime. Own CUDA kernels never run. |
| Access | Root SSH; `jibo-skill` (uid 2000) is in group `video`, which owns the GPU nodes. |
| Headroom (eye and vision running) | About 600 MB of RAM, no swap; about one core of CPU; GPU about 57 % busy on average (9 to 90 %) at 396 MHz of a possible 852 MHz. |
| Thermal | Idle 47.5 to 51.5 °C; games reached 52 to 53.5 °C. Lowest trip about 62 °C (PLL sensor). |
| Initial limits (engineering, not hardware) | One inference worker; about 200 MiB of added memory for a first prototype; one small GPU batch in flight; sub-millisecond dispatches as a target to investigate. |

The project path, the SSH destination and the exact launcher invocation are not in the
handoff. Phase 0 recovers them from the owner's workspace; this document does not invent them.

## Budget: what fits

### Parameters and memory

Counted from the two models' `config.json` on Hugging Face. Both have 24 layers: 18 Gated
DeltaNet (16 heads of 128 dimensions for keys and values, a 4-tap convolution) and 6 gated full
attention (8 query heads and 2 key-value heads of 256 dimensions, rotary on a quarter of each
head). Both have a vocabulary of 248,320 tokens and tied embeddings.

| | Qwen3.5-2B (v21's torso) | Qwen3.5-0.8B |
|---|---|---|
| Hidden size / MLP width | 2048 / 6144 | 1024 / 3584 |
| Embedding table (lookup only) | 509 M | 254 M |
| Layer parameters (read by every forward) | 1,373 M | 498 M |
| ... of which MLP | 906 M | 264 M |
| ... of which Gated DeltaNet projections | 379 M | 190 M |
| ... of which attention projections | 88 M | 44 M |
| Layer weights, 8-bit with group scales (~8.5 bits per weight) | ~1.46 GB | ~0.53 GB |
| Layer weights, 4-bit with group scales (~4.5 bits per weight) | ~0.77 GB | ~0.28 GB |
| Matmul work per token (2 × layer parameters) | ~2.7 GFLOP | ~1.0 GFLOP |

Runtime state beyond the weights is small for both. The recurrent state is 16 × 128 × 128 fp32
per Gated DeltaNet layer, 1 MiB, so 18 MiB in all. The attention cache is 24 KiB per token
(6 layers, keys and values, 2 heads of 256, fp32), so 12 MiB at 512 tokens. Activations for a
prefill chunk of 64 tokens are a few MiB. One saved prefix snapshot for the shared-state path
costs about the recurrent state plus the cache, roughly 30 MiB. The tokenizer tables are an
estimate of 15 to 25 MB, to be measured.

Estimated resident set for the 0.8B decider at 4 bits with a 512-token window: about 280 MB
of weights, 20 MB of tokenizer and about 70 MB of state and scratch, so **about 370 MB plus
whatever the GL driver allocates for a context**. That fits inside the 600 MB of headroom. It
does not fit the handoff's initial 200 MiB prototype limit, which was set for Needle; the owner
has since set the limit at about 400 MB ([Decisions](#decisions)). The 2B decider at 4 bits
would need about 870 MB and cannot run. At about 2.5 bits per weight it would squeeze into
roughly 520 MB, but that leaves no margin and costs accuracy that has not been measured.

On flash, the 0.8B artifact is about 0.28 GB of layer weights plus the embedding table: about
0.5 GB in fp16 or 0.25 GB in 8-bit. Free space on the robot's writable partition is not in
the handoff, so Phase 0 checks it.

### Where the weights live

There is no swap, so anonymous memory can never be reclaimed and an over-budget process
invites the OOM killer into the robot's own services. The CPU backend therefore memory-maps
the weight file read-only. File-backed pages can be dropped and re-read under pressure, which
costs latency but cannot exhaust memory. The GPU backend copies the weights into GL buffers,
which the driver pins, and then releases the CPU mapping (`madvise(MADV_DONTNEED)`) so the
weights are not resident twice. The embedding table always stays memory-mapped. Each token
touches one row (2 KiB in fp16 at 0.8B), so its resident cost is a few pages.

### Compute

These are estimates from vendor peak figures, not measurements, and they are only for
planning. The handoff warns against reading the GPU's idle share as guaranteed capacity, and
these ranges are wide for that reason.

- **CPU.** A Cortex-A15 core peaks at about 8 fp32 FLOP per cycle, about 15 GFLOPS at 1.94 GHz.
  A NEON kernel that dequantises weights on the fly might sustain 5 to 10 GFLOPS. With one core
  to spare, a 128-token prompt takes about **13 to 26 s** on 0.8B and 35 to 70 s on 2B. ARMv7
  has no int8 dot-product instruction (`SDOT` is ARMv8.2). An int8 path through
  `VMULL.S8`/`VPADAL` might gain 1.5 to 2 times, but it changes the numerics and comes later.
- **GPU.** GK20A has 192 lanes at 2 FLOP per FMA, so it peaks at 152 GFLOPS at 396 MHz and
  327 GFLOPS at 852 MHz. The clock belongs to the governor, which we do not change. Kepler has
  no fast fp16 arithmetic, so the shaders compute in fp32. If a hand-written GLSL kernel holds
  25 to 50 % of peak and the renderer keeps its share, the effective rate is perhaps 20 to
  80 GFLOPS. A 128-token prompt would then take about **2 to 6 s** on 0.8B.
- **Memory bandwidth** (Tegra K1 LPDDR3, about 15 GB/s by specification, shared with the
  display) is not the limit for prefill. Each layer's weights are read once per forward and
  used across all of the prompt's tokens. It does mean that our traffic competes with the
  renderer's.

The Gated DeltaNet recurrence and the attention scores add about 2 % to the matmul work at
these lengths. The final layer is full attention, and only the `<answer>` and option positions
are read from it. Its output projection and MLP can therefore run on those K + 1 rows alone,
which saves about 3 % of the forward for free.

What this means for use: a decision costs seconds. The port suits decisions a robot can make
in the background or while it is already speaking or moving. It does not suit a turn-taking
reflex that needs an answer in 100 ms. Several questions about one state share that state's
encoding, so asking three questions about an utterance costs less than three forwards. With
short states the saving is smaller, because each question carries its own options in about
50 to 80 tokens.

## Design

### Components

```
                         jibo-skill (Node.js)                    owner's Jibo workspace
                                 │ JSON over a Unix socket        (run.sh shim, sysroot,
                                 ▼                                  ABI check, deploy)
 ┌───────────────────────────── jibo-decider (C99, one process) ─────────────────────────────┐
 │ socket thread ── queue (depth 1) ──► worker thread (owns the model and the GL context)     │
 │                                        │                                                   │
 │   render (prompting.py) ─► fit + tokenise (_fit, byte-level BPE) ─► forward driver ─► head │
 │                                                                       │            (fp32)  │
 │                                                  ┌────────────────────┴────────┐           │
 │                                                  │ backend op interface        │           │
 │                                                  ├─────────────┬───────────────┤           │
 │                                                  │ CPU (scalar,│ GL 4.3 compute│ (CUDA     │
 │                                                  │ NEON)       │ (dlopen libGL)│  later)   │
 │                                                  └─────────────┴───────────────┘           │
 │ model.jdw (mmap): config, 4-bit layer weights, fp16/int8 embeddings, fp32 head, temps      │
 │ tokenizer.jdt (mmap): vocabulary, merges, special tokens                                   │
 └────────────────────────────────────────────────────────────────────────────────────────────┘
        ▲ produced on the workstation by
 ports/jibo/tools/export_jdw.py  (this repository; reads a checkpoint through StrandsDeciderModel)
```

### Language, toolchain and ABI

- **C99**, which GCC 4.8.4 compiles fully, with NEON intrinsics from `arm_neon.h`. There is no
  C++, so there is no libstdc++ to collide with the robot's `GLIBCXX_3.4.20`. There is no Rust
  either, which avoids the handoff's Rust toolchain question.
- Built in the existing `jibo-armcc` image against the `work/fs/p2` sysroot, through the
  owner's existing wrapper. `ports/jibo/scripts/build-jibo.sh` takes the image, the sysroot and
  the wrapper as environment variables and stops if any is unset.
- Flags: `-march=armv7-a -mfpu=neon -mfloat-abi=hard`, with no `-ffast-math`. The handoff lists
  VFPv3. Cortex-A15 cores normally also report `vfpv4`, which adds fused multiply-add, so check
  `/proc/cpuinfo` before using `-mfpu=neon-vfpv4`. Fused multiply-add rounds differently, so
  switching to it is a numerical change in its own step.
- The same sources also build for x86-64 with the host compiler, using the scalar kernels.
  Parity tests and continuous integration run there, so most of the work needs no robot.
- **Vendored, pinned dependencies:** `utf8proc` (MIT) for NFC normalisation and the Unicode
  categories the pre-tokenizer needs, and a small JSON parser (cJSON, MIT) for the socket
  protocol. Both are C99. Licences are recorded in `THIRD_PARTY_NOTICES.md`.
- `check-jibo-abi.sh` wraps the owner's existing checker. It also asserts that the binary is
  ELF32 ARM with the hard-float attribute and the expected loader, that `NEEDED` lists only
  libc, libm, libpthread, libdl and the loader, that no symbol version is above `GLIBC_2.21`,
  and that no `GLIBCXX` or `CXXABI` version appears at all.

### The weight file (`.jdw`)

Written by `ports/jibo/tools/export_jdw.py` on the workstation, from any checkpoint
`StrandsDeciderModel.load` accepts (a local directory or a Hub id):

- A fixed header: magic, format version, and the length of a JSON metadata block. The JSON
  holds the torso config, `max_length`, `pointer_dim`, `temperature`, `temperature_by_kind` and
  `ordinal_smoothing`. It also holds the provenance: the checkpoint id and revision,
  `base_model` and `base_revision`, the repository commit, and the SHA-256 of every source file.
  A tensor table gives each tensor's name, shape, dtype, quantisation group size and offset.
- **LoRA folded in fp32** before any rounding, `W + (alpha / r) B A`, as `mlx_engine.merge_lora`
  does. That function's refusals are kept: no DoRA, no `modules_to_save`, no per-module ranks.
  Doing the merge in fp32 avoids the bf16 rounding that accounts for most of MLX's 0.0138
  difference ([inference.md](inference.md#serving-on-a-mac-with-mlx)).
- Layer weights in one of three formats, chosen at export: `f32` for parity, `q8` (int8 with
  an fp16 scale per group of 32) and `q4` (packed 4-bit with an fp16 scale and, if it is
  needed, an offset per group of 32). Norm weights, the convolution taps, `A_log` and
  `dt_bias` stay fp32. The embedding table is `f16` or `q8`. The pointer head is fp32. The
  multi-token-prediction layer and the vision tower are not exported.
- Every tensor starts on a 64-byte boundary, so the file can be memory-mapped and uploaded
  without repacking.
- The tokenizer is compiled to `tokenizer.jdt`: the vocabulary as a sorted string table, the
  merges as a hash from pair to rank, the special and added tokens, and the normaliser and
  pre-tokenizer settings read from `tokenizer.json`. The exporter refuses settings that the C
  tokenizer does not implement, rather than ignore them.

### Tokenizer and prompt rendering

The port is exact or it is wrong. A prompt that tokenises differently is a different input,
and the pointer head reads specific token positions.

- `render.c` is a line-for-line port of `render_content`, `_option_block`, `render_question` and
  `render_state`, including collapsing whitespace in option descriptions and the `—` separator.
  Option spans are byte offsets into the rendered UTF-8. Inputs are NFC-normalised before
  rendering, so spans and token offsets share one coordinate system. A structured (JSON) state
  is rendered as `json.dumps(indent=2, ensure_ascii=False)` would render it. Only strings come
  first; structured states follow once the golden tests cover them.
- `tokenizer.c` implements byte-level BPE with the pre-tokenizer split that `tokenizer.json`
  declares, and keeps each token's byte offsets. It ports `_fit` exactly: the question's
  reserve, the question's front truncation, the state's truncation (whichever end the
  tokenizer's `truncation_side` cuts, pinned by the golden tests), whether special tokens are
  added, and `strict_window`. It also ports `_option_token_index`, including the error raised
  when truncation has removed an option.
- The acceptance criterion is **zero mismatches**, in token ids and in option indices, against
  Hugging Face `tokenizers`. The comparison corpus is every JevBench prompt, the
  `device_parity.py` and `bench_local.py` requests, a robot-style prompt set, and generated
  adversarial strings (combining marks, emoji, CJK, mixed scripts, long runs of whitespace and
  digits).

### Forward driver and backend interface

`forward.c` is backend-agnostic. It walks the 24 layers in the order `layer_types` gives and
calls a small op interface on opaque tensor handles. Each backend implements:

| Op | Notes |
|---|---|
| `embed` | The CPU gathers rows from the memory-mapped table, dequantised to fp32, and uploads T × H. |
| `rmsnorm`, `rmsnorm_gated` | Qwen3.5 RMSNorm; the gated form multiplies by `silu(z)` inside the DeltaNet block. |
| `matmul_q` | fp32 activations times `f32`/`q8`/`q4` weights. The output can be restricted to a row subset for the final layer. |
| `conv1d_causal_silu` | 4-tap depthwise convolution over the DeltaNet's q, k, v channels, carrying 3 steps of state. |
| `gdn_scan` | l2-normalised q and k, the decay from `A_log`, `dt_bias` and `a`, the gate `b`, and the gated delta rule over the chunk's tokens, carrying the fp32 recurrent state. |
| `attention` | Per-head q and k RMSNorm, rotary on the first 64 of 256 dimensions (for text, all three M-RoPE sections share one position, so this is ordinary RoPE), GQA 8 to 2, causal softmax against the cache, and the sigmoid output gate. |
| `swiglu`, `add` | Elementwise. |
| `state_save`, `state_restore` | Copy the recurrent, convolution and attention state, for the shared prefix. |
| `read_rows` | Copy K + 1 final hidden vectors to the host. This is the only readback in a forward. |

The CPU backend starts with the recurrent form of the delta rule, token by token, which is
exact and simple. The chunked form (the one `mps_kernels.py` accelerates) is an optimisation
for later, with its own parity check. That file's note on the Neumann-series inverse applies
here too: any triangular solve gets tested on real, correlated, l2-normalised keys, not only
on random inputs.

The prefill runs in chunks of tokens (64 to start) to bound activation memory. The state
crosses chunk boundaries exactly as it crosses the prefix-to-suffix boundary, so the
chunk-boundary tests and the prefix tests are the same tests.

**Shared prefix, sequentially.** For N > 1 questions the driver encodes the state, saves its
state once, and then, for each question, restores the saved state and forwards that
question's suffix. This is `_slot_probs_shared_prefix` with a batch of one per question. It
keeps memory flat in N, at the cost of the batch parallelism the desktop gets. A single
question forwards the whole prompt, as `evaluate` does.

**Head and answers** run on the CPU in fp32 for every backend: LayerNorm, the q and k
projections (256 wide), the scaled dot products, the per-kind temperature, the masked softmax,
and then `derive_confidence` and `derive_score_confidence` with the ordinal-smoothing floor
correction, rounded to 4 places as `_to_answer` rounds.

### GL compute backend

- **Context.** The worker thread creates its own desktop GL 4.3+ context and owns it for the
  life of the process. It reuses the display access and the library shim that the game host's
  `run.sh` arranges. Phase 0 recovers exactly what that script does, because `DISPLAY=:0` alone
  may not reproduce it. Nothing is injected into the renderer or into vision.
- **Shaders.** `#version 430` compute shaders using SSBOs, shared memory and `barrier()`, with no
  warp-shuffle assumptions. The quantised GEMM unpacks 4-bit or 8-bit weights and their fp16
  scales (`unpackHalf2x16`) in registers and accumulates in fp32. Tile sizes come from the
  limits the probe queries, such as shared memory per workgroup and maximum invocations.
- **Bounded dispatches.** Kepler cannot preempt a running compute dispatch at fine grain. A
  long dispatch therefore holds back the eye's next frame, which at about 55 FPS has an 18 ms
  budget. The driver slices every op into dispatches of a bounded amount of work, tuned from
  measurement towards the handoff's sub-millisecond target. At 128 tokens, one 0.8B MLP
  projection is about 0.94 GFLOP, roughly 10 to 50 ms at the estimated rates, so it is split
  into many tiles. A forward is then on the order of a few thousand dispatches. The driver
  inserts a fence every few slices, keeps one small batch in flight, and waits with
  `glClientWaitSync` and a real timeout. It never spins on a zero timeout.
- **Visibility and completion are separate.** The driver issues
  `glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)` between dependent dispatches,
  `GL_BUFFER_UPDATE_BARRIER_BIT` before a readback with `glGetBufferSubData`, and fences for
  completion.
- **Zero copy on unified memory.** Tegra K1's CPU and GPU share DRAM. Persistently mapped
  buffers (`glBufferStorage` with `MAP_PERSISTENT` and `MAP_COHERENT`, core in GL 4.4) may let
  the CPU write embeddings and read the K + 1 rows without copies. Coherent mappings can be
  uncached on the CPU side, though, so this is measured against plain `glBufferSubData` before
  it is used.
- **Residency.** Weights, the recurrent and attention state, the prefix snapshot and scratch all
  live in GL buffers for the life of the process. A forward uploads T × H of embeddings and
  reads back (K + 1) × H. The intermediate step of Phase 7, matmuls on the GPU with the rest on
  the CPU, reads back once per layer. That is acceptable as a stepping stone but not as the
  final design.
- **Cancellation.** A host deadline is checked between slices. Abandoning a request stops
  submitting work and waits for the slice in flight. Nothing in GL cancels a running kernel,
  and the design does not pretend otherwise.

### CUDA as a later backend

The op interface admits a CUDA Driver API backend: `cuInit`, a context created with
`CU_CTX_SCHED_BLOCKING_SYNC`, and `sm_32` cubins or old-driver-compatible PTX loaded with
`cuModuleLoadDataEx`, with the JIT logs captured. That backend needs a CUDA 6.5-era toolkit for
`sm_32`, or hand-written PTX, and the robot has neither configured. It also competes with
`jibo-lps-service`'s per-frame CUDA work. It starts only if the GL backend stalls on a driver
limitation. A device-query executable that compiles no kernels is cheap, and it can run
alongside Phase 5 to settle what the driver reports.

### The service

- `jibo-decider` runs as `jibo-skill`, which is in group `video` and so can open the GPU nodes,
  from an isolated directory that the deploy step creates and the cleanup step removes.
- It listens on a **Unix domain socket** with file permissions, not TCP. The desktop server binds
  127.0.0.1 with no authentication, and a robot on a home network should not copy that.
- Its protocol is the JSON of `POST /v1/systemone`, one request and one response per
  connection, with the schema of `schema.py`. Skills write the same request a Strands agent
  writes (compare `examples/strands/_client.py`), and the published request and response
  shapes stay the one contract. `GET /health` becomes a `health` message that returns the
  checkpoint, the model's provenance, the backend and the temperatures.
- **One worker and a queue of one.** A request that arrives while one runs and one waits gets
  an immediate `busy` error, and the skill decides what to do. The context window defaults to
  512 tokens on the robot, configurable up to the checkpoint's `max_length`. Requests are
  truncated, or with `strict_window` refused, exactly as on the desktop.
- **A governor between requests and between slices.** It pauses new work above a temperature
  threshold below the lowest trip point, read from the thermal zones read-only, and above a
  memory floor read from `MemAvailable`. It also caps the duty cycle (GPU-busy time per
  minute). The thresholds come from Phase 8's measurements and the owner's agreement. Starting
  proposals: pause at 58 °C on the PLL sensor, or when `MemAvailable` falls below 400 MB.
- `jibo-decider ask` mirrors `strands-decider ask`, for tests and the benchmarks.

### Code layout (proposed)

```
ports/jibo/
  README.md               build, deploy and cleanup instructions; the pins (the handoff's JIBO_PORT.md)
  runtime/                C99: jdw.c tokenizer.c render.c fit.c forward.c head.c answer.c
                          backend_cpu.c kernels_neon.c backend_gl.c shaders/*.comp
                          server.c cli.c; vendor/utf8proc, vendor/cjson
  tools/
    export_jdw.py         checkpoint -> model.jdw + tokenizer.jdt (+ manifest with SHA-256)
    dump_golden.py        reference tensors, token ids, option indices, probabilities, answers
    jibo-gl-probe.c       the handoff's first GPU deliverable, standalone
    bench-jibo.c, bench-jibo.sh   op and end-to-end benchmarks; coexistence runs
  scripts/
    build-jibo.sh         cross build in jibo-armcc; paths from the environment, never defaults
    check-jibo-abi.sh     wraps the owner's checker; the assertions above
  tests/                  pytest: build the runtime with the host compiler, compare with the torch engine
  results/                logs, hashes, timings, observed limits (one directory per run)
```

The Python tools live outside `src/strands_decider`, as `evaluation/` does, so the published
package and its `mypy` profile do not change. Robot-specific glue, such as the shim, the
launcher, SSH targets and the sysroot, stays in the owner's Jibo workspace and is passed in.

## Numerical parity

Change one numerical behaviour at a time, and report absolute and relative errors rather than
expecting bitwise identity. Each stage is compared with the stage before it, on fixed request
sets: the 54 answers of `evaluation/device_parity.py` (both engine paths, states of about 40 to
3,000 tokens, all three question types), a robot-style set of short states, and JevBench for
the stages that can change accuracy.

| Stage | Compared against | Expected | Gate |
|---|---|---|---|
| 1. C runtime, `f32` weights, x86 | torch engine, fp32, CPU | float-order differences only | per-layer hidden states within 1e-4 relative; probabilities within 1e-4; no answer changes; shared-prefix and whole-prompt paths agree |
| 2. `q8` weights, x86 | stage 1 | quantisation error | no answer changes on the 54; max probability difference at most 0.02 (MLX's bf16 merge gives 0.0138) |
| 3. `q4` weights, x86 | stage 1 | quantisation error | thresholds written down before the run: proposed, JevBench answers agree on at least 97 % of tasks and Brier score worsens by at most 0.01 |
| 4. ARMv7 NEON (qemu, then the robot) | stage 2 or 3 on x86, same file | op order, NEON | probabilities within 1e-5 |
| 5. GL backend (robot) | stage 4, same file | GPU float order | probabilities within 1e-4; no answer changes |

Stage 1 runs in continuous integration on a tiny random Qwen3.5 with a pointer head, built the
way `tests/test_vision.py` and `tests/test_mlx_engine.py` build theirs, so the suite still
downloads nothing. The real checkpoints run on the workstation. The golden dumps record every
layer's output at the option and `<answer>` positions and at a sample of others, so a failure
names the first layer that diverges.

Calibration: the temperatures were fitted on the unquantised model. Stages 2 and 3 report ECE
on the calibration split. If quantisation worsens calibration measurably, the temperatures are
refitted on the quantised model's logits, which `jibo-decider` can emit raw. The refit is
written to the `.jdw` metadata with a note in `results/`. It never happens silently.

## Plan

The phases are ordered by dependency. Phase 5 does not depend on the model and can start at
once, in parallel with Phases 1 to 4. Phase 9 (training the 0.8B decider) is also parallel:
Phases 1 to 4 develop against the 2B checkpoint on x86, where memory is no object, and against
the tiny model in CI. Every step that runs something on the robot happens inside the owner's
agreed deployment scope ([Coexistence and safety rules](#coexistence-and-safety-rules)).

### Phase 0: recover the workspace and pin everything (workstation only)

- Locate `jibo-armcc`, `work/fs/p2`, the game host's `run.sh` and its shim, and the ABI checker.
  Record what `run.sh` sets up for GL: the display, the libraries, environment variables and
  the user.
- Check the free space on the robot's writable partition (read-only), and decide where
  artifacts go.
- Pin the repository commit, the checkpoint and its Hub revision, `base_model` and
  `base_revision`, the Python environment (torch 2.7.1 and transformers 5.17.0, as in
  [inference.md](inference.md#environments-for-serving)), the `jibo-armcc` image digest,
  and the versions of `utf8proc` and cJSON. Record SHA-256 for every input file.
- **Exit:** `ports/jibo/README.md` with the pins, and the recovered launcher facts written
  down.

### Phase 1: reference artifacts (this repository, x86)

- `export_jdw.py` (`f32` first) and `dump_golden.py`, with tests on the tiny random model.
- The tokenizer comparison corpus and its expected ids and offsets.
- **Exit:** the `.jdw` round-trips (reading it back reproduces the merged torch weights
  exactly), and the golden set is committed for the tiny model and stored with hashes in
  `results/` for the real checkpoints.

### Phase 2: the C runtime in fp32 on x86

- Rendering, tokenizer and `_fit`; the forward driver and the CPU backend (scalar); the head and
  the answers; `jibo-decider ask`.
- **Exit:** parity stage 1 passes on the tiny model in CI and on the 2B checkpoint on the
  workstation, on both engine paths. The tokenizer has zero mismatches on the corpus.

### Phase 3: quantisation on x86

- `q8`, then `q4`, as export options and backend kernels. Stages 2 and 3 run on 2B now and are
  repeated on 0.8B when it exists.
- **Gate G1:** does the chosen format keep the decider's answers and calibration, and does the
  0.8B artifact's resident set fit the budget the owner sets? If `q4` fails stage 3, the
  fallback is `q8` for attention and DeltaNet with `q4` MLP, then all `q8` (about 0.53 GB, which
  only fits if the owner accepts it).

### Phase 4: ARMv7 CPU build (the first half of the immediate milestone)

- NEON kernels for `matmul_q`, the norms and the elementwise ops. Single-threaded, matching the
  one inference worker.
- `build-jibo.sh` and `check-jibo-abi.sh`. Functional tests under `qemu-arm` with the
  `work/fs/p2` sysroot: the tiny model fully, and a few real prompts.
- On the robot, inside the agreed scope: the first run as `jibo-skill` from an isolated
  directory, recording resident memory, latency and CPU use. The 2B checkpoint does not fit,
  so this run uses the 0.8B decider, or a truncated test model until that exists.
- **Exit:** an ABI-clean binary, parity stage 4, and the first measured CPU latency.

### Phase 5: the GL probe (parallel; the second half of the milestone)

- `jibo-gl-probe.c` as the handoff specifies. It creates its own context through the recovered
  launcher arrangement, queries `GL_MAX_COMPUTE_WORK_GROUP_*`, shared memory, SSBO limits and
  `GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT`, compiles the `values[i] = 3u * i + 7u` shader,
  dispatches 4 × 64, and checks all 256 values. It logs the compiler and linker output and the
  GL errors, records device time (`GL_TIME_ELAPSED` queries if they are exposed) and host
  latency, and uses the proper barriers and a blocking fence wait.
- It also measures how long a no-op dispatch takes, then a 1 ms busy dispatch while the eye
  animates, and records the eye's frame-time tails if the renderer exposes them.
- **Exit:** a correct dispatch of our own on the robot's GPU, with logs in `results/`.

### Phase 6: one real operation, CPU against GPU (completes the milestone)

- The 0.8B MLP up-projection (1024 to 3584, `q4`) at 128 tokens, with real weights from the
  `.jdw`, on both backends. The timing includes preparation, upload and readback, and the
  dispatch slicing is swept.
- **Gate G2:** is the GPU faster than the CPU for this op, counting transfers, without
  degrading eye frame times beyond what the owner accepts? If not, the port ships CPU-only
  and Phase 7 stops.

### Phase 7: the GPU forward

1. Matmuls on the GPU, everything else on the CPU, with a readback per layer. This is a
   stepping stone, to check the op order on real data.
2. All ops resident on the GPU, with one readback per forward, and the prefix snapshot on the
   GPU.
3. The variants the handoff asks to compare: GPU prefill of the state with CPU suffixes, and
   the reverse, as well as everything on the GPU.

- **Exit:** parity stage 5, latency recorded at 64, 128, 256 and 512 tokens with 1 and 3
  questions.

### Phase 8: the service and coexistence

- `jibo-decider` over the Unix socket, the governor, and a minimal Node skill client.
- The handoff's benchmark matrix: idle, listening, speaking, and moving or face tracking, each
  without and then with inference. For each, record end-to-end p50, p95 and p99 latency, CPU
  use, total added memory (including GL), temperatures and clocks, eye frame-time tails, audio
  faults, and vision latency where they can be observed. Anything that cannot be observed is
  marked unavailable.
- **Gate G3:** is the end-to-end latency useful for the owner's decisions, with no unacceptable
  degradation? The governor's thresholds are set from these runs.

### Phase 9: a decider small enough to run (parallel, research)

- A preregistration, as `CONTRIBUTING.md` requires for any training run:
  `research/preregistrations/PREREGISTRATION-v22.md`, stating predictions and failure conditions
  before training. The run is the v21b recipe (`configs/experiments/v21b.yaml`) with
  `base_model: "Qwen/Qwen3.5-0.8B-Base"` and nothing else changed. The LoRA targets name the
  same modules, so they carry over.
- Measured on JevBench and the robot-style set, then exported and taken through parity stages
  2 and 3.
- Only if accuracy or budget demand it, a second preregistered run truncating the torso (for
  example the first 16 of 24 layers, keeping the 3 : 1 DeltaNet-to-attention pattern) and
  retraining the adapter and head. This saves a third of the memory and compute again.
- **Exit:** a published or owner-held 0.8B checkpoint with its JevBench result recorded next to
  v21's, whether or not it meets its predictions.

### First pull requests

To keep each change small, as `CONTRIBUTING.md` asks, with an issue first since the work is
significant:

1. This document.
2. `ports/jibo/tools/export_jdw.py` and `dump_golden.py`, with tests on the tiny model (Phase 1).
3. The tokenizer and the renderer in C, with the zero-mismatch test (Phase 2, first half).
4. The CPU forward and the head, with the stage-1 parity test in CI (Phase 2, second half).
5. `jibo-gl-probe.c` (Phase 5). This one can go first; it is independent.

## Coexistence and safety rules

These are inherited from the handoff and are binding on every phase:

- Copying or running anything on the robot, even in `/tmp`, is a deployment step, and only
  happens within the owner's agreed scope. Use `jibo-skill` where that is enough. Keep
  artifacts in one isolated directory and remove them explicitly.
- Do not flash firmware, overwrite system libraries, stop the vision, audio, body or render
  services, change clocks, touch the fan or the thermal controls, or add swap. The runtime never
  commands the motors.
- Do not keep persistent GPU kernels or deep queues: bounded dispatches only, with at most one
  small batch in flight. Short dispatches are a scheduling courtesy, not a real-time
  guarantee.
- Measure the whole robot, not just the process. Stop adding load at the first sign of service
  degradation or unexpected resource pressure, and record it in `results/`.
- Promise no latency or throughput before Phase 6 and Phase 8 have measured them.

## Risks

| Risk | Effect | Mitigation |
|---|---|---|
| The 0.8B decider loses too much accuracy | The port works but answers worse | Phase 9 measures it before the runtime depends on it; the runtime is torso-size agnostic, so a better small torso drops in |
| Memory over budget (GL driver overhead unknown) | OOM pressure on robot services | File-backed weights on the CPU path; Phase 4 and Phase 6 measure real totals; the governor's `MemAvailable` floor |
| Coarse GPU preemption | Eye stutters | Bounded dispatches, fences, a duty-cycle cap; Gate G2 can rule the GPU out |
| Old GL driver (R21.5) miscompiles or limits compute shaders | Wrong results or crashes | The probe first; every shader has a CPU twin and a parity test; no exotic GLSL |
| Tokenizer mismatch on rare text | Silent wrong answers | Zero-mismatch gate on a broad corpus; the exporter refuses unsupported tokenizer settings |
| Quantisation hurts calibration | Confidence bands mislead | ECE reported at stages 2 and 3; temperatures refitted only openly |
| Thermal rise during multi-second GPU work | Throttling, or a trip near the 62 °C sensor | Temperature-gated governor; sustained runs in Phase 8 before any skill depends on it |
| Seconds of latency is not useful for the owner's decisions | Low value | Gate G3; [Alternatives considered](#alternatives-considered) lists a LAN fallback |

## Decisions

Decided by the owner on 7 October 2026:

1. **Target: this repository's decider**, not `needle-rs`.
2. **Memory: about 400 MB at most** for the whole decider process, GL allocations included.
   This replaces the handoff's initial 200 MiB. The 0.8B estimate of about 370 MB plus the GL
   driver's share leaves little margin, so Phase 4 and Phase 6 measure the real total first,
   and the governor's floor is set against this ceiling.
3. **Train a 0.8B decider only if it costs under $15** of Hugging Face credits.
   [PREREGISTRATION-v22.md](../research/preregistrations/PREREGISTRATION-v22.md) records the
   run, its budget and the bar the port applies to its result.
4. **The code lives in `ports/jibo/`** in this repository.
5. **Latency: under 2 s per decision** is the starting target. The estimates above put a
   128-token prompt on the 0.8B forward at 2 to 6 s on the GPU, so the target is not yet
   expected to hold. Phase 6 measures it. If it misses, the levers in order are GPU kernel
   efficiency, shorter prompts from the skills, and a truncated torso under its own
   preregistration.

Still open: the **deployment scope** for Phases 4 to 8 (which robot, which directory, which
user, and the cleanup procedure), and access to the Jibo workspace for Phase 0.

## Alternatives considered

- **llama.cpp (ggml).** It implements Qwen3.5's hybrid architecture and quantisation, and it
  can return final hidden states. But current releases need a C++17 compiler, so we would need
  a second, newer cross toolchain with a statically linked libstdc++ audited against the ABI
  ceiling. Its GPU backends (CUDA for modern toolkits, Vulkan, OpenCL) do not match this
  driver, so the GL backend would have to be built inside ggml's backend model anyway. It
  stays useful as a third reference on x86 for spot checks.
- **ncnn or another general runtime.** Its GPU path is Vulkan, which this firmware does not
  offer. Qwen3.5's Gated DeltaNet layers would need custom operators.
- **Keeping the 2B torso with 2-bit or 3-bit weights.** It just about fits in memory, leaves no
  margin, triples the compute against 0.8B, and has an unmeasured accuracy cost. It is not
  pursued unless Phase 9 fails badly.
- **Qwen3-0.6B (all full attention).** It is supported by the older code path, but the
  research found Qwen3.5 about 0.10 better than Qwen3 at matched size
  ([history.md](../research/history.md#what-the-torso-knows-untrained)).
- **Not native: a LAN call** from a skill to `strands-decider serve` on a desktop. It answers
  in about 115 ms on an RTX 3090 with the full 2B model. This is outside the handoff's
  mission, but it is the cheapest route to good decisions on the robot whenever the network
  is there. A skill could prefer it and fall back to the native runtime offline, through the
  same JSON contract.
