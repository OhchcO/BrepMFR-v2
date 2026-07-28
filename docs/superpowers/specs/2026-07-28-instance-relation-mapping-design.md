# Instance Relation Mapping Design

## Goal

Make the existing instance-relation attention bias selectable between the
current signed mapping and a positive-only mapping without adding a model
branch, loss, encoder, or training stage.

## Mapping Contract

- `signed` is the default and preserves every current relation value.
- `positive_only` applies `clamp_min(0)` to the current signed relation.
- Dynamic positive relations retain the existing confidence weighting.
- Oracle same-instance values remain positive one; different-instance values
  change from negative one to zero.
- Padding, diagonal masking, gate scale, schedule, injection layers, affinity
  loss, and gradient isolation remain unchanged.

## Configuration

The canonical option is `instance_relation_mapping` with choices `signed` and
`positive_only`. It is saved in checkpoint hyperparameters and test result
provenance. Old checkpoints default to `signed`.

The PowerShell experiment launcher exposes
`-InstanceRelationMapping signed|positive_only`, passes it to training, and
adds `_positive` to the experiment label when positive-only is selected. Thus
`-Mode oracle -InstanceRelationMapping positive_only` writes under an
`oracle_positive` result name without changing the underlying oracle mode.

## Validation

Tests must prove signed backward compatibility, positive-only removal of
negative dynamic and Oracle relations, invalid-value rejection, configuration
propagation, result provenance, launcher parameter exposure, and positive
result naming.
