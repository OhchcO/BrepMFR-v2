# Residual Dynamic Instance Bias Design

## Goal

Revise only the existing instance-relation branch so that it implements true
multi-task representation learning and uses predicted instance relations as a
safe residual attention bias. The original BrepMFR input representation,
semantic head, dataset preparation, BIN format, and experiment aliases remain
unchanged.

The method must support the causal comparison:

```text
baseline
  -> true multi-task affinity supervision
  -> true multi-task supervision plus residual dynamic instance bias
  -> oracle residual instance bias
```

## Existing Failure Mode

The current `late_gated` implementation predicts affinity from a detached
mid-encoder representation:

```text
P = affinity_head(detach(H_mid))
```

Consequently, the affinity loss trains the affinity head but cannot teach the
shared encoder to represent instance structure. The predicted relation is then
added directly to the attention logits in the final layers. Prediction errors
therefore perturb the only semantic-attention path.

This design makes it difficult to distinguish three effects:

1. learning instance-aware shared features;
2. predicting pairwise instance affinity;
3. injecting the prediction into semantic attention.

## Selected Architecture

### True multi-task affinity prediction

Dynamic affinity logits are predicted from the non-detached middle encoder
state:

```text
P_logits = affinity_head(H_mid)
```

For `dynamic_affinity`, the training objective is:

```text
L_total = L_semantic + lambda_affinity * L_affinity(P_logits, P_target)
```

The affinity loss therefore updates both the affinity head and the shared
encoder representation.

### Prediction-guidance gradient decoupling

The relation used by the semantic attention branch is derived from detached
affinity logits:

```text
B_instance = relation_mapping(detach(P_logits))
```

Thus:

- affinity-loss gradients update the affinity head and shared encoder;
- semantic-loss gradients through the bias path update the residual gate;
- semantic-loss gradients through the bias path do not alter the affinity
  predictor merely to make semantic classification easier.

For `dynamic`, which has no affinity loss, the relation is not detached. This
preserves a semantic-loss learning signal for its affinity predictor and keeps
the existing `dynamic_no_loss` ablation meaningful.

### Confidence-weighted positive relation

The existing relation mappings remain supported. The primary experiment uses
`positive_only`, where uncertain and predicted different-instance pairs are
neutral and only confident predicted same-instance pairs receive positive
bias:

```text
centered = 2 * sigmoid(P_logits) - 1
relation = max(centered * abs(centered) ** confidence_power, 0)
```

Padding pairs and diagonal pairs remain masked.

### Zero-initialized residual biased attention

Each configured late encoder layer computes an ordinary attention branch and a
biased attention branch using the same layer parameters:

```text
Y_base   = Attention(X, B_graph)
Y_biased = Attention(X, B_graph + B_instance)
Y        = Y_base + gate * (Y_biased - Y_base)
```

The residual gate is a learnable scalar per injected encoder layer. Every gate
is initialized to exactly zero and is bounded during use. Therefore a newly
initialized model starts with output exactly equal to the ordinary BrepMFR
attention path.

The residual combination applies to the self-attention result before the
layer's ordinary residual connection, normalization, and feed-forward block.
Both attention calls share projections and all other parameters. No second
network is introduced.

`bias_scale` continues to scale the instance relation before the biased
attention call. A zero bias scale makes the two attention branches identical
and therefore disables the residual contribution.

## Experiment Alias Contract

- `baseline`: original BrepMFR; no affinity head and no instance-bias branch.
- `dynamic_no_loss`: dynamic relation prediction without affinity loss. The
  semantic objective may train the predictor through the residual biased
  attention path.
- `affinity_only`: non-detached affinity prediction and affinity supervision;
  all residual gates are fixed at zero so there is no semantic bias injection.
- `instance`: non-detached affinity prediction and affinity supervision;
  semantic bias uses a detached predicted relation and learnable residual
  gates.
- `oracle`: ground-truth instance relation through the same residual biased
  attention path; no predicted affinity logits.

The existing user-facing mode names and PowerShell commands remain valid.
No geometry or representation experiment is changed.

## Configuration and Compatibility

- Existing raw data and BIN caches remain valid.
- No new dataset field is required.
- The existing late-layer count, confidence power, relation mapping, bias
  scale, delay, and ramp configuration remain available.
- The residual gate default is exactly zero for new runs.
- Existing result directories and checkpoints are not modified.
- Old checkpoints using the previous direct late-gated implementation are
  retained for historical evaluation but are not architecture-compatible with
  new residual-bias training runs.
- New runs must use new result directories and must be retrained.

## Observability

Training logs continue to record affinity loss, affinity weight, and bias
progress. The reported instance-bias gate becomes the mean effective residual
gate across injected late layers. Standalone evaluation continues to support
the existing bias-scale override so the same checkpoint can be evaluated with
the residual bias enabled and disabled.

## Tests

Focused unit tests must prove:

1. affinity loss in `dynamic_affinity` propagates through affinity logits to
   the middle node representation;
2. semantic loss through the predicted bias does not update the affinity head
   in `dynamic_affinity`;
3. semantic loss can update the predictor in `dynamic`;
4. a zero-initialized residual gate produces exactly the baseline layer output;
5. a nonzero residual gate changes the output when instance bias is nonzero;
6. `affinity_only` retains affinity logits and shared-feature gradients while
   keeping semantic bias disabled;
7. residual bias is used only in the configured final encoder layers;
8. existing aliases preserve their dataset and representation switches;
9. existing CLI and PowerShell commands remain syntactically valid.

Only focused unit tests, compilation, and lightweight synthetic forward and
backward checks are authorized. Training and full dataset conversion are
excluded.

## Acceptance Criteria

- All focused tests pass.
- The baseline forward path is unchanged.
- All four instance experiment aliases remain launchable through the existing
  scripts.
- No training or full dataset conversion is performed during implementation.
- The code exposes enough logged state to compare `affinity_only`, `instance`,
  and same-checkpoint bias-disabled evaluation.
