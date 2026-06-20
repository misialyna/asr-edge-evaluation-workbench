# ADR-0001: Faster-Whisper as the v1 baseline runtime

- Status: Accepted
- Date: 2026-06-20
- Deciders: project owner + senior mentor review

## Context

We need a single, mature, edge-friendly ASR runtime for v1. The
project's hard constraints are:

1. Must run on aarch64 Linux (NVIDIA Jetson AGX Orin).
2. Must support at least one INT8 / FP16 quantization path so we
   can compare quality / latency / resource trade-offs.
3. Must be importable from Python without a heavyweight
   training-style dependency.
4. Must be license-compatible with a public portfolio repo
   (prefer permissive, no GPL).

## Decision

Use **Faster-Whisper** (CTranslate2-based) as the v1 baseline
runtime. It supports `int8`, `int8_float16`, `float16`, and
`float32` compute types, has a clean Python API, runs on aarch64
Linux including Jetson, and is MIT-licensed.

## Consequences

- We are bound to its compute_type taxonomy and to its
  decoding-parameter surface. Other Whisper runtimes
  (Whisper.cpp, TensorRT-optimized variants) will need adapters
  in v2.
- The WhisperTRT integration from NVIDIA is intentionally
  **out of scope for v1** to keep the deliverable focused.

## Alternatives considered

- `openai/whisper`: slower, no INT8 weights, no first-class
  edge story. Rejected.
- `whisper.cpp`: not Python-first; would require a C++/Python
  shim. Rejected for v1.
- TensorRT-LLM Whisper / WhisperTRT: powerful but premature for
  a one-week MVP and adds nontrivial build complexity. Deferred
  to v2 (see ADR-0006 in backlog).
