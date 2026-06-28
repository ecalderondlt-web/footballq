# Experiment Protocol

Scientific artifacts must record split manifest path and hash, feature view,
objective mode, command/config provenance, and alignment status.

Default alignment is period-aware:

```text
(match_id, period, frame_t)
sample_id = "{match_id}:{period}:{frame_t}"
```

Index-order or `(match_id, frame_t)` alignment is non-scientific and must be
enabled explicitly with `allow_legacy_alignment=True`.

Possession and availability probes are leakage sanity checks for full-state
representations. Raw global x displacement uses geometric names, not tactical
progression names.
