"""YAML experiment-config loader and validator.

This module is intentionally thin: it does the I/O and delegates
validation to :mod:`asr_edge_eval.schemas`. A failure to load a config
always surfaces as :class:`ConfigError` so callers don't need to catch
``yaml.YAMLError``, ``FileNotFoundError`` and ``ValidationError``
separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schemas import ExperimentConfig

__all__ = ["ConfigError", "load_experiment", "dump_experiment"]


class ConfigError(ValueError):
    """Raised when an experiment YAML cannot be loaded or validated."""


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    ExperimentConfig
        A validated, fully-typed experiment specification.

    Raises
    ------
    ConfigError
        If the file does not exist, is not valid YAML, or fails schema
        validation.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ConfigError(f"Experiment config not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Experiment config {p} must be a YAML mapping at the top level, got {type(raw).__name__}"
        )
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        # Surface a single, readable error string. Pydantic v2 errors
        # are already human-friendly; we just strip the trailing
        # technical line for a single-line message and keep the full
        # error in the chained exception.
        raise ConfigError(f"Invalid experiment config {p}:\n{exc}") from exc


def dump_experiment(config: ExperimentConfig, path: str | Path) -> None:
    """Write an experiment config back to YAML.

    Useful for round-trip tests and for tooling that mutates
    configurations programmatically.
    """
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = config.model_dump(mode="json")
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
