# Lint Debt

This file records deferred repo-wide Ruff debt so the integrity branch does not
claim to be lint-clean prematurely.

Last checked: 2026-07-03

Command:

```bash
python -m ruff check . --statistics
```

Current result:

```text
36 E501  line-too-long
4  I001  unsorted-imports
2  UP035 deprecated-import
42 total errors
6 auto-fixable
```

Files currently affected:

| File | Count | Codes |
| --- | ---: | --- |
| `scripts/eval_latent_flow.py` | 1 | E501 |
| `scripts/run_decoder_learning_curve.py` | 1 | E501 |
| `scripts/run_latent_flow_ablation.py` | 1 | E501 |
| `scripts/sample_latent_flow.py` | 2 | I001, E501 |
| `src/footballq/decoding/eval.py` | 1 | E501 |
| `src/footballq/decoding/learning_curve.py` | 12 | E501 |
| `src/footballq/decoding/models.py` | 2 | E501 |
| `src/footballq/decoding/stress.py` | 5 | E501 |
| `src/footballq/decoding/suite.py` | 2 | UP035, E501 |
| `src/footballq/decoding/train.py` | 1 | E501 |
| `src/footballq/latent_flow/ablation.py` | 5 | I001, UP035, E501 |
| `src/footballq/latent_flow/baselines.py` | 1 | E501 |
| `src/footballq/latent_flow/eval.py` | 1 | E501 |
| `src/footballq/latent_flow/models.py` | 1 | E501 |
| `tests/test_decoder_learning_curve.py` | 1 | E501 |
| `tests/test_decoder_train_smoke.py` | 1 | E501 |
| `tests/test_latent_flow_dataset.py` | 2 | I001, E501 |
| `tests/test_latent_flow_train_smoke.py` | 1 | E501 |
| `tests/test_probe_labels.py` | 1 | I001 |

Policy for this sprint:

- Touched integrity-sprint files should remain Ruff-clean.
- Do not report the repository as lint-clean until `python -m ruff check .`
  passes.
- If lint cleanup is deferred again, update this file with the latest exact
  counts and command output.
