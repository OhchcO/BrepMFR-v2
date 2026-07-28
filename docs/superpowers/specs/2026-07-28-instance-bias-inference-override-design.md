# Instance Bias Inference Override Design

## Goal

Allow one trained checkpoint to be evaluated twice with its saved instance
bias scale or with an explicitly overridden inference-only scale, especially
zero, without changing checkpoint weights or training configuration.

## Interface

- Python evaluator: optional `--instance-bias-scale-override FLOAT`.
- PowerShell wrapper: optional nullable
  `-InstanceBiasScaleOverride FLOAT`.
- Omitting the option preserves the checkpoint-saved scale.
- Values must be finite and non-negative; zero disables the injected bias.

## Data Flow

After `BrepSeg.load_from_checkpoint`, the evaluator reads
`model.brep_encoder.instance_relation_bias.bias_scale`. It records that
checkpoint value, applies the optional override directly to the loaded
in-memory module, and records the effective value used for evaluation.

`test_metrics.json` and `test_log.txt` contain:

- `checkpoint_instance_bias_scale`
- `effective_instance_bias_scale`
- `instance_bias_scale_overridden`

To prevent the second evaluation from replacing the first, an explicit scale
override with no `--output-dir` writes to a checkpoint-relative directory named
`test_bias_scale_<value>`. Evaluation without an override keeps the historical
checkpoint-directory output behavior, and an explicit `--output-dir` takes
priority.

## Compatibility

No training code, BIN cache, checkpoint state, or default evaluation behavior
changes. Baseline checkpoints also accept the override for a uniform command
interface, although their relation bias is inactive.

## Testing

Unit tests cover omitted overrides, explicit zero, positive values, negative
and non-finite rejection, and missing model structure. A PowerShell parser
test protects wrapper syntax.
