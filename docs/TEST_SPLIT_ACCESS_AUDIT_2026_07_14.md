# Test-Split Access Audit: 2026-07-14

## Finding

The legacy TD-JEPA trainer instantiated train, validation, and test sharded datasets for every run
and automatically wrote `embeddings_sample.pt` from the first test batch after training. Several
historical documents correctly state that no test loss or falsification metric was computed, but
incorrectly strengthen that statement to say the test split was completely untouched.

Known affected PFF studies include the original 100-update transfer diagnostic, the 2,000-update
transfer repeat, the balanced-curriculum comparison, and the square-root sampler comparison. Their
run directories contain the automatic embedding artifact.

## Scope

The embedding export happened after optimization, validation, checkpoint writing, and checkpoint
selection. It performed a forward pass only and did not update model weights, optimizer state,
validation metrics, gate thresholds, or selected checkpoints. The reported validation gates remain
valid as validation results, but the historical PFF test split cannot be described as unread or
untouched at the tensor-file level.

## Remediation

- sharded training now instantiates only configured tensor splits
- `training.validation_split` is explicit and may be disabled
- `training.embedding_sample_split` is opt-in and defaults to disabled
- run manifests record loaded tensor splits plus validation and embedding split settings
- train-only manifests are supported without synthetic validation/test tensors
- the position-only PFF study uses a physically separate train/validation-only projection with zero
  test shards
- regression tests verify that excluded test shards are not loaded and default embedding export is
  disabled

Future claims must distinguish “no test metric” from “no test tensor access.” A clean confirmatory
test study requires a separately frozen protocol and should not reuse the historically exposed split
as if it were pristine.
