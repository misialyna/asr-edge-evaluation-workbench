# ADR-0002: Mozilla Common Voice Polish as the v1 evaluation set

- Status: Accepted
- Date: 2026-06-20
- Deciders: project owner + senior mentor review

## Context

The v1 deliverable needs a frozen, license-clean, well-documented
Polish speech corpus with reference transcripts. Candidates
considered:

| Dataset | Pros | Cons | Verdict |
|---|---|---|---|
| **Mozilla Common Voice Polish v11.0** | CC0, large, well-cited, easy Hugging Face `datasets` integration, sentence + audio fields, validated split. | Many clips are short; demographic coverage varies. | **Selected**. |
| PolEval Polish ASR task | Curated, official Polish evaluation. | Smaller; some clips gated by registration; license per-clip varies. | Backup, deferred to v2. |
| "MC Speech Dataset" (initial brief) | Referenced in the original brief. | Not a widely-citable Polish ASR resource; harder to defend in an interview. | **Dropped**. |

## Decision

Use **Mozilla Common Voice Polish v11.0** as the v1 evaluation
set. The frozen subset is 20 clips (2–10 s each) drawn from the
`validated` split, normalized to 16 kHz mono WAV, and written
to ``data/prepared/audio/`` alongside a JSONL manifest at
``data/manifests/benchmark_manifest.jsonl``.

The subset size is small on purpose: v1 is a one-week MVP, the
goal is to demonstrate a *reproducible evaluation*, not to
produce a state-of-the-art WER number.

## Consequences

- WER numbers reported in v1 are not directly comparable to
  numbers from PolEval or other benchmarks. We say so
  explicitly in the README and the report.
- The pipeline is dataset-agnostic: a v2 release can swap in
  PolEval by writing a new manifest.
- The audio corpus is **not** committed to git. It is downloaded
  on demand by ``data/scripts/prepare_polish_eval.py``. The
  manifest is committed so the manifest version is stable.

## What we deliberately do **not** claim

- We do not claim coverage of regional Polish dialects.
- We do not claim coverage of noisy, accented, or code-switched
  speech. The selected clips are clean read speech.
- We do not claim demographic balance. Common Voice's
  demographic metadata is patchy and we do not filter on it.
