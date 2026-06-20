"""Rule-based scenario recommendation.

The recommender turns a list of :class:`BenchmarkRun` rows into a
:class:`Recommendations` value that names a configuration for each
of the three deployment scenarios:

* **real-time** — RTFx must be ≥ 1.5; among those, the lowest WER wins.
* **balanced** — minimize a weighted score combining normalized WER,
  shortfall of RTFx below 1.0, and normalized RAM peak.
* **quality-first** — the lowest WER wins, regardless of speed.

The weights are loaded from a YAML file (``configs/recommend_weights.yaml``)
or supplied directly. They are part of the result object so the
recommendation is never a black box: every number in the rationale
can be re-derived from the runs and the weights.

The function is total: empty input raises :class:`ValueError`; one
configuration produces a recommendation for all three scenarios
pointing at the same config.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

import yaml

from ..schemas import BenchmarkRun, ConfigSummary, Recommendations, RecommendationWeights

__all__ = ["recommend", "summarize_runs", "load_weights"]


def load_weights(path: str | Path | None = None) -> RecommendationWeights:
    """Load recommendation weights from YAML, falling back to defaults."""
    if path is None:
        return RecommendationWeights()
    p = Path(path)
    if not p.is_file():
        return RecommendationWeights()
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"recommend_weights YAML must be a mapping; got {type(raw).__name__}")
    return RecommendationWeights.model_validate(raw)


def summarize_runs(runs: list[BenchmarkRun]) -> list[ConfigSummary]:
    """Aggregate ``runs`` into one :class:`ConfigSummary` per config_id.

    Error rows (``error`` not None) are excluded from quality and
    latency aggregates; they are still counted in ``num_runs`` so
    the operator can see how many attempts failed.
    """
    groups: dict[str, list[BenchmarkRun]] = defaultdict(list)
    for r in runs:
        groups[r.config_id].append(r)

    out: list[ConfigSummary] = []
    for cid, rs in groups.items():
        ok = [r for r in rs if r.error is None]
        wers = [r.quality.wer for r in ok if math.isfinite(r.quality.wer)]
        rtfxs = [r.latency.rtfx for r in ok if r.latency.rtfx > 0]
        ram_peaks = [r.resources.ram_peak_mb for r in ok if r.resources.ram_peak_mb is not None]
        gpu_mem_peaks = [
            r.resources.gpu_mem_peak_mb for r in ok if r.resources.gpu_mem_peak_mb is not None
        ]
        temp_peaks = [
            r.resources.temperature_peak_c for r in ok if r.resources.temperature_peak_c is not None
        ]
        power_peaks = [r.resources.power_peak_w for r in ok if r.resources.power_peak_w is not None]
        out.append(
            ConfigSummary(
                config_id=cid,
                wer_mean=fmean(wers) if wers else float("inf"),
                wer_std=pstdev(wers) if len(wers) > 1 else 0.0,
                rtfx_mean=fmean(rtfxs) if rtfxs else 0.0,
                rtfx_std=pstdev(rtfxs) if len(rtfxs) > 1 else 0.0,
                ram_peak_mb=max(ram_peaks) if ram_peaks else None,
                gpu_mem_peak_mb=max(gpu_mem_peaks) if gpu_mem_peaks else None,
                temperature_peak_c=max(temp_peaks) if temp_peaks else None,
                power_peak_w=max(power_peaks) if power_peaks else None,
                num_runs=len(rs),
            )
        )
    return out


def _normalize(values: list[float], *, invert: bool) -> list[float]:
    """Min-max normalize to [0, 1]. Invert flips the ranking direction.

    Used to bring WER (lower is better) and RAM (lower is better)
    to the same scale as the 0..1 weights. The "invert" flag is
    not used here; kept for clarity in case future scenarios
    require "higher is better" inputs.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    if invert:
        return [(hi - v) / (hi - lo) for v in values]
    return [(v - lo) / (hi - lo) for v in values]


def _balanced_score(
    summary: ConfigSummary,
    *,
    all_summaries: list[ConfigSummary],
    weights: RecommendationWeights,
) -> float:
    """Compute the balanced score for one configuration.

    Lower is better. Components are normalized WER, RTFx shortfall
    below 1.0, and normalized RAM peak. The score is a weighted
    sum of those three components, with the weights normalized so
    they always sum to 1.
    """
    wers = [s.wer_mean for s in all_summaries if math.isfinite(s.wer_mean)]
    rtfxs = [s.rtfx_mean for s in all_summaries if s.rtfx_mean > 0]
    rams = [s.ram_peak_mb for s in all_summaries if s.ram_peak_mb is not None]

    norm_wer = 0.0
    if wers:
        lo, hi = min(wers), max(wers)
        if hi - lo > 1e-12:
            norm_wer = (
                (summary.wer_mean - lo) / (hi - lo) if math.isfinite(summary.wer_mean) else 1.0
            )
        else:
            norm_wer = 0.0

    shortfall = max(0.0, 1.0 - summary.rtfx_mean)  # 0 if already >1
    norm_rtfx = 0.0
    if rtfxs:
        lo, hi = min(rtfxs), max(rtfxs)
        if hi - lo > 1e-12:
            norm_rtfx = (summary.rtfx_mean - lo) / (hi - lo)  # higher is better
        # else leave as 0
    # Convert the "higher is better" rtfx to a penalty:
    penalty_rtfx = 1.0 - norm_rtfx
    penalty_rtfx += shortfall  # always penalize sub-1 RTFx
    if penalty_rtfx < 0:
        penalty_rtfx = 0.0
    if penalty_rtfx > 1:
        penalty_rtfx = 1.0

    norm_ram = 0.0
    if summary.ram_peak_mb is not None and rams:
        lo, hi = min(rams), max(rams)
        if hi - lo > 1e-12:
            norm_ram = (summary.ram_peak_mb - lo) / (hi - lo)
        else:
            norm_ram = 0.0
    elif summary.ram_peak_mb is None:
        # No RAM data: treat as neutral (0.5) so the score isn't
        # silently biased low.
        norm_ram = 0.5

    w_total = weights.wer + weights.rtfx + weights.ram
    if w_total <= 0:
        w_wer, w_rtfx, w_ram = 1 / 3, 1 / 3, 1 / 3
    else:
        w_wer = weights.wer / w_total
        w_rtfx = weights.rtfx / w_total
        w_ram = weights.ram / w_total

    return w_wer * norm_wer + w_rtfx * penalty_rtfx + w_ram * norm_ram


def _real_time_score(summary: ConfigSummary) -> float:
    """Real-time priority: RTFx must be ≥ 1.5; lower WER wins.

    Returned as a tuple ``(eligible, score)``. A non-eligible
    config has score ``inf`` so it never wins.
    """
    if summary.rtfx_mean < 1.5:
        return float("inf")
    if not math.isfinite(summary.wer_mean):
        return float("inf")
    return summary.wer_mean


def _quality_first_score(summary: ConfigSummary) -> float:
    return summary.wer_mean if math.isfinite(summary.wer_mean) else float("inf")


def recommend(
    runs: list[BenchmarkRun],
    *,
    weights: RecommendationWeights | None = None,
    weights_path: str | Path | None = None,
) -> Recommendations:
    """Compute scenario-based recommendations.

    Parameters
    ----------
    runs:
        The result rows from a single experiment.
    weights:
        Optional explicit weights. If not given, ``weights_path`` is
        consulted; if that is also ``None``, the defaults are used.
    """
    if not runs:
        raise ValueError("no runs to recommend from")
    if weights is None:
        weights = load_weights(weights_path)
    summaries = summarize_runs(runs)
    if not summaries:
        raise ValueError("no non-empty summaries to recommend from")

    # Real-time: pick the eligible config with the lowest WER.
    rt_ranked = sorted(summaries, key=_real_time_score)
    if _real_time_score(rt_ranked[0]) == float("inf"):
        # No config met the RTFx threshold. Fall back to the fastest
        # one and surface that explicitly in the rationale.
        rt_winner = max(summaries, key=lambda s: s.rtfx_mean)
        rt_fallback = True
    else:
        rt_winner = rt_ranked[0]
        rt_fallback = False
    rt_score = _real_time_score(rt_winner)

    # Balanced: minimum weighted score.
    bal_ranked = sorted(
        summaries,
        key=lambda s: _balanced_score(s, all_summaries=summaries, weights=weights),
    )
    bal_winner = bal_ranked[0]
    bal_score = _balanced_score(bal_winner, all_summaries=summaries, weights=weights)

    # Quality-first: minimum WER.
    q_ranked = sorted(summaries, key=_quality_first_score)
    q_winner = q_ranked[0]
    q_score = _quality_first_score(q_winner)

    rationale = _build_rationale(
        rt_winner=rt_winner,
        rt_fallback=rt_fallback,
        bal_winner=bal_winner,
        q_winner=q_winner,
        weights=weights,
    )

    return Recommendations(
        real_time_config_id=rt_winner.config_id,
        balanced_config_id=bal_winner.config_id,
        quality_first_config_id=q_winner.config_id,
        real_time_score=rt_score if math.isfinite(rt_score) else 0.0,
        balanced_score=bal_score,
        quality_first_score=q_score if math.isfinite(q_score) else 0.0,
        weights=weights,
        summaries=summaries,
        rationale=rationale,
    )


def _build_rationale(
    *,
    rt_winner: ConfigSummary,
    rt_fallback: bool,
    bal_winner: ConfigSummary,
    q_winner: ConfigSummary,
    weights: RecommendationWeights,
) -> str:
    parts: list[str] = []
    if rt_fallback:
        parts.append(
            f"No configuration met the RTFx≥1.5 real-time threshold; "
            f"falling back to the fastest available: {rt_winner.config_id} "
            f"(mean RTFx={rt_winner.rtfx_mean:.2f})."
        )
    else:
        parts.append(
            f"{rt_winner.config_id} qualifies for real-time "
            f"(mean RTFx={rt_winner.rtfx_mean:.2f} ≥ 1.5) with the "
            f"lowest WER ({rt_winner.wer_mean:.3f})."
        )
    parts.append(
        f"{bal_winner.config_id} minimises the weighted score "
        f"(wer={weights.wer:.2f}, rtfx={weights.rtfx:.2f}, "
        f"ram={weights.ram:.2f})."
    )
    parts.append(
        f"{q_winner.config_id} has the lowest WER "
        f"({q_winner.wer_mean:.3f}), suitable for batch / quality-first use."
    )
    return " ".join(parts)
