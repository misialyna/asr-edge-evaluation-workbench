# ADR-0005: Recommendation logic is rule-based, not learned

- Status: Accepted
- Date: 2026-06-20
- Deciders: project owner + senior mentor review

## Context

After every benchmark run, the system must pick one
configuration per scenario ("real-time", "balanced",
"quality-first"). The temptation is to call this an "AI
recommendation"; that is exactly the wrong move for a
junior-portfolio project, because:

1. A learned ranker trained on three configurations is
   overkill and the reviewer will see through it.
2. A black-box scorer is **not** auditable, and the entire
   point of a benchmarking project is that every number
   is auditable.
3. The owner must be able to explain the recommendation in
   an interview in under a minute.

## Decision

The recommendation is a **transparent, rule-based scorer**
with explicit, configurable weights. The weights are stored
in `configs/recommend_weights.yaml` and printed in the
report. The three scenarios are:

* **Real-time**: configuration with `RTFx ≥ 1.5` and the
  lowest WER. If none qualify, fall back to the fastest
  available and surface that explicitly in the rationale.
* **Balanced**: minimize
  `w_wer * norm(wer) + w_rtfx * penalty(rtfx) + w_ram * norm(ram_peak_mb)`,
  with weights from the YAML.
* **Quality-first**: configuration with the lowest WER,
  irrespective of speed.

## Consequences

- The "weights live in a YAML" rule means a reviewer can
  re-run the analysis with a different weighting and see
  the recommendation change accordingly. This is the
  single most useful property of the system.
- The scorer is covered by unit tests that assert known
  outputs for known inputs.
- The rationale string is included in the
  `Recommendations` object and rendered into the report.
- We do **not** ship a learned ranker in v1. A v2
  "LTR-style" model trained on the accumulated runs is
  interesting but is not on the v1 critical path.
