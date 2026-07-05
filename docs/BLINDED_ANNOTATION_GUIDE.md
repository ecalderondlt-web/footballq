# Blinded Annotation Guide

This guide is for human review of diagnostic football tracking clips. The goal is
to label what is visible in the clip without seeing cluster IDs, residual scores,
or positive/control status.

These annotations are not tactical evidence by themselves. They become useful
only after they are compared against matched controls and after the probe and
discovery gates are reviewed.

## Files To Use

Use only:

- `annotator/annotations.csv`
- GIF files referenced by the `clip_path` column

Do not open:

- `private/annotation_key.csv`
- any run summary containing cluster IDs, residual scores, or control status

## How To Annotate

For each row, watch the GIF and write exactly one label in the `annotation`
column.

Allowed labels:

- `tactical_pattern`: coordinated movement, spacing, pressure, transition, or
  ball/player interaction that appears football-meaningful from the clip alone
- `routine_motion`: ordinary smooth movement with no clear motif
- `tracking_artifact`: missing players/ball, identity jump, impossible motion,
  rendering issue, or provider/tracking artifact
- `ambiguous`: not enough visual evidence to decide

Use the labels literally. Do not add notes in the `annotation` cell; free-form
text makes the enrichment script harder to interpret. The analysis script
rejects labels outside this list.

## Review Rules

- Judge only what is visible in the clip.
- Do not infer from file order, blind ID, match ID, or period.
- Do not try to identify clusters or residual ranking.
- Mark tracking or rendering problems as `tracking_artifact`, even if the clip
  also looks interesting.
- Use `ambiguous` when a clip could plausibly fit multiple labels.

## After Annotation

Run package validation again, allowing filled annotation cells:

```bash
python scripts/validate_blinded_annotation_package.py \
  --annotator-csv runs/diagnostics/v2_context_w0p05_slot_recon_margin_blinded_balanced_seed7_h02/annotator/annotations.csv \
  --key-csv runs/diagnostics/v2_context_w0p05_slot_recon_margin_blinded_balanced_seed7_h02/private/annotation_key.csv \
  --manifest-json runs/diagnostics/v2_context_w0p05_slot_recon_margin_blinded_balanced_seed7_h02/render_manifest.json \
  --allow-filled-annotations
```

Then summarize the annotation result:

```bash
python scripts/analyze_blinded_annotations.py \
  --annotator-csv runs/diagnostics/v2_context_w0p05_slot_recon_margin_blinded_balanced_seed7_h02/annotator/annotations.csv \
  --key-csv runs/diagnostics/v2_context_w0p05_slot_recon_margin_blinded_balanced_seed7_h02/private/annotation_key.csv \
  --manifest-json runs/diagnostics/v2_context_w0p05_slot_recon_margin_blinded_balanced_seed7_h02/render_manifest.json \
  --positive-labels tactical_pattern \
  --out runs/diagnostics/v2_context_w0p05_slot_recon_margin_blinded_balanced_seed7_h02/annotation_summary.json
```

Interpret the summary as diagnostic until the full integrity gate passes.
