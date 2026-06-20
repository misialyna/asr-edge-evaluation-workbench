# ADR-0004: tegrastats over nvidia-smi for Jetson telemetry

- Status: Accepted
- Date: 2026-06-20
- Deciders: project owner + senior mentor review

## Context

We need a reliable source of per-run CPU, RAM, GPU, GPU
memory, temperature, and (when present) power measurements
on the Jetson AGX Orin. Two candidates:

* `nvidia-smi` — works on data-center GPUs, **partially
  supported on Jetson** (no power, no thermals, intermittent
  GPU memory). It is the right tool for an x86 box; it is
  the wrong tool for a Jetson.
* `tegrastats` — ships on every Jetson image; reports RAM,
  swap, per-core CPU, GPU percent and MiB, temperature, and
  power (when the INA231 sensor is present, e.g. on Orin).

## Decision

Use `tegrastats` as the **primary** Jetson telemetry source.
`nvidia-smi` may be used as a debug supplement on x86 only.

The collector spawns `tegrastats` with a configurable
sampling interval, redirects stdout to a per-run log file
under `results/runs/telemetry/`, and parses the lines in
a background thread. After the inference, the parsed lines
are aggregated into a `ResourceMetrics` row.

## Consequences

- The parser must be tolerant. JetPack 5 and JetPack 6 emit
  *slightly* different line formats (temperature prefix,
  GPU % vs GPU MiB). The parser is regex-based and yields
  `None` for any field that is not present, so a missing
  sensor never crashes the run.
- `tegrastats` is spawned as a child process group so the
  collector can stop it reliably with `SIGTERM` (and
  `SIGKILL` as a fallback) without leaving a zombie
  process behind.
- For demo mode on a developer machine, the collector
  silently falls back to `psutil` for CPU/RAM and
  `None` for everything else. The report explicitly
  flags missing resource data as such.

## See also

- `docs/metrics.md` — exact definitions of every field.
- `tests/unit/test_tegrastats_parser.py` — fixture logs.
