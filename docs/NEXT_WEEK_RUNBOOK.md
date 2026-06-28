# Next Week Runbook

1. Build TD-JEPA examples with `future_nonoverlap_context_only`,
   `geometry_only`, and `--split-manifest`.
2. Train representation v2 using the manifest split.
3. Export embeddings with period-aware sample IDs.
4. Build probes, decoders, latent rollouts, and discovery datasets with the same
   split manifest.
5. Run falsification and discovery controls before any tactical interpretation.
6. Treat residual-ranked clips as blinded diagnostics only; keep annotator files
   separate from key files until matched controls are reviewed.

Required verification:

```bash
python -m pytest -q
python -m ruff check <touched files>
```

Repo-wide `python -m ruff check .` is not yet a clean gate because older
untouched files still have pre-existing lint debt. Do not report it as a sprint
failure unless the scope expands to final repo-wide lint cleanup.
