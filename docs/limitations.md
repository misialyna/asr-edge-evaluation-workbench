# Limitations

A portfolio project that hides its limitations is not a
portfolio project worth reading. This page lists the things
v1 **does not** do, in one place, so a reviewer does not
have to read between the lines of the report.

## Scope limitations

1. **One runtime.** v1 uses Faster-Whisper. NVIDIA's
   TensorRT-optimized Whisper, `whisper.cpp`, and the
   upstream `openai/whisper` are all out of scope. A v2
   adapter behind the `AsrRuntime` Protocol is planned.
2. **One dataset.** v1 uses a 20-clip frozen subset of
   Mozilla Common Voice Polish. PolEval and other Polish
   resources are not evaluated.
3. **Three configurations.** Only `small/int8`, `small/fp16`,
   and `medium/fp16` are compared. `large-v3`,
   `distil-large-v3`, and any beam-search variants are
   deferred.
4. **Offline inference only.** No streaming, no
   chunked-decoding, no VAD pre-filtering. The VAD flag is
   present in the config but is `false` for v1.
5. **No fine-tuning.** The model weights are the upstream
   CTranslate2 conversions; no Polish fine-tuning is
   performed in v1.
6. **No speaker diarization, no TTS, no LLM.** All
   explicitly out of scope.

## Methodological limitations

1. **Small subset.** 20 clips is enough to demonstrate a
   reproducible evaluation but not enough to claim
   statistical significance. A v1.1 expansion to 50–100
   clips is in the backlog.
2. **No cross-device comparison.** v1 does not run the
   same workload on x86 + GPU and an Apple Silicon laptop
   to put the Jetson numbers in context. Deferred to v2.
3. **No energy-per-audio-hour metric.** Power data is
   collected when present, but a normalized "kWh per
   hour of audio" is not yet computed.
4. **Single language.** Polish only. The pipeline is
   language-agnostic but v1 is not validated on a second
   language.
5. **Greedy decoding.** `beam_size=1` is used for all
   measurements to keep latency comparable. Higher
   beam sizes will produce lower WER at the cost of
   higher latency, and are not measured.

## Recommendation limitations

1. **Rule-based, not learned.** The recommender is a
   weighted score. It is fully auditable but it does
   not learn from prior runs.
2. **Three scenarios.** Real-time, balanced, and
   quality-first. A deployment that needs a fourth
   (e.g. "lowest possible peak power") will have to
   define its own weights.
3. **No confidence intervals.** Per-config standard
   deviations are reported, but the report does not
   include bootstrap confidence intervals on the
   recommendation. Easy to add in v1.1.

## Software limitations

1. **Faster-Whisper must be installed separately** as the
   `asr` extra. A developer machine running `make demo`
   does not need it; a Jetson running a real benchmark
   does.
2. **The dashboard requires `streamlit` and `plotly`**, in
   the `dashboard` extra. The CLI and the report do not.
3. **No GPU support on macOS MPS in v1.** The
   `FasterWhisperAdapter` uses `device="auto"`, which
   defers to CTranslate2's device selection. Apple
   Silicon MPS is on the v2 backlog.
4. **No multi-process orchestration.** A single Python
   process runs the whole benchmark. There is no
   task-queue, no worker pool, no Kubernetes.

## What this means for the report

The numbers in `results/reports/benchmark_report.md` are
**point estimates** from a 20-clip, single-runtime,
three-configuration measurement on a specific Jetson AGX
Orin image. They are not a leaderboard entry. They are
evidence that the pipeline produces reproducible,
auditable results — which is the actual deliverable.
