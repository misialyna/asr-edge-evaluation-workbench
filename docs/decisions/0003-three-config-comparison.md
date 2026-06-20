# ADR-0003: Comparing three Faster-Whisper configurations in v1

- Status: Accepted
- Date: 2026-06-20
- Deciders: project owner + senior mentor review

## Context

v1 needs a small, defensible set of configurations to compare.
Too few (one) is uninteresting; too many (six or more) makes
the one-week MVP unrealistic and inflates the chance of
unreproducible results.

## Decision

Compare exactly three configurations of Faster-Whisper:

1. **`small` + `int8`** — the most efficient configuration
   on the menu; expected to be the fastest and the lightest
   on memory, with a measurable quality hit.
2. **`small` + `float16`** — the same model as (1) with
   weights kept at FP16 precision; isolates the effect of
   INT8 quantization from model size.
3. **`medium` + `float16`** — larger model at FP16; the
   quality ceiling at the cost of higher latency and memory.

The pair (1) vs. (2) answers "what does INT8 cost us?"; the
pair (2) vs. (3) answers "what does going to a larger model
buy us?". Together they cover the three scenarios the README
will recommend against.

## Consequences

- Greedy decoding (`beam_size=1`) is used for all three
  configurations to keep latency comparisons fair and
  deterministic. ``vad_filter`` is disabled for the same
  reason. Decoding options beyond that are empty.
- "Real-time" here is operational, not strict. We report
  RTFx ≥ 1.5 as the eligibility threshold for the
  real-time scenario, but a real deployment may pick a
  tighter bound.
- We do not test `large-v3` or `distil-large-v3` in v1 —
  both add non-trivial model weight downloads and push
  beyond the time budget.
