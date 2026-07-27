# Late-Gated Dynamic Instance Bias Design

## Goal

Improve only the existing instance-relation branch so that the original
BrepMFR baseline remains the sole baseline and the existing experiment aliases
remain the user-facing training and evaluation interface.

The target claim is that `instance` (dynamic bias plus affinity supervision)
improves semantic segmentation over `baseline`. Cylindrical positions, face
point clouds, dataset conversion, semantic heads, and the original BrepMFR
geometry path are outside this change.

## Existing Failure Mode

The current implementation predicts affinity from the input node embedding and
adds a non-negative bias to the shared attention bias before every Transformer
layer. Consequently:

- uncertain predictions still increase attention;
- predicted different-instance pairs cannot reduce attention;
- noisy early predictions affect all representation levels;
- affinity-loss gradients update the shared semantic representation;
- the negative class combines easy different-semantic pairs with the important
  same-semantic/different-instance pairs.

## Selected Design

New training runs launched through the existing instance aliases use a
`late_gated` strategy. Direct checkpoints or configurations without the new
strategy field retain the legacy behavior for compatibility.

### Affinity prediction

The affinity head consumes a detached mid-encoder representation. This prevents
the auxiliary affinity loss from changing the shared early semantic features.
The head continues to use the existing content, topological-distance, geometric-
distance, and angle descriptors.

For dynamic modes:

```text
probability = sigmoid(logit)
centered_relation = 2 * probability - 1
confidence = abs(centered_relation) ** confidence_power
relation = confidence * centered_relation
```

This produces a signed relation in `[-1, 1]`: confident same-instance pairs
receive positive bias, confident different-instance pairs receive negative
bias, and uncertain pairs receive approximately zero bias.

### Gated late-layer injection

Affinity is predicted once from the middle of the executed encoder stack. The
bias is injected only into the final configurable number of encoder layers.

```text
effective_scale =
    alias_bias_scale
    * warmup_progress
    * max_gate_scale
    * sigmoid(learnable_gate_logit)
```

The learnable gate is initialized to `0.05`, bounded by `max_gate_scale`, and
learned by the semantic objective. The relation tensor is detached before
semantic attention injection in `dynamic_affinity`; therefore semantic loss
updates the gate but not the affinity predictor. In `dynamic`, semantic loss
may update the affinity predictor because that ablation has no affinity loss.

The first `instance_bias_delay_epochs` use zero injected bias. It then increases
linearly over `instance_bias_ramp_epochs`. Validation during training uses the
same progress as the current epoch. Standalone inference uses full progress.

### Structured affinity supervision

The upper-triangular valid pairs are divided into three groups:

1. same-instance positives;
2. same-semantic/different-instance hard negatives;
3. different-semantic/different-instance easy negatives.

The loss averages each non-empty group and then averages the groups. This is a
deterministic balanced `1:1:1` weighting without stochastic pair sampling.
Semantic labels and instance labels are used only during training.

## Existing Alias Contract

- `baseline`: original BrepMFR, no affinity head and no instance bias.
- `dynamic_no_loss`: late-gated dynamic bias without affinity supervision.
- `affinity_only`: affinity supervision with bias scale zero.
- `instance`: late-gated dynamic bias plus structured affinity supervision.
- `oracle`: ground-truth signed instance relation through the same late-gated
  injection path.

No new primary experiment alias is introduced. Existing PowerShell training
and test commands remain valid. Additional instance-bias parameters are
optional launcher parameters with fixed defaults.

## Compatibility

- Existing BIN files remain valid; no geometry fields are added.
- Old checkpoints without `instance_bias_strategy` instantiate the legacy path.
- New alias-launched checkpoints record all new parameters in hyperparameters.
- Existing result directories are not overwritten because the user controls
  launch time and result naming as before.

## Tests

Unit tests must prove:

- centered confidence suppresses uncertain relations and signs confident pairs;
- a zero alias scale produces zero attention bias while retaining logits;
- dynamic-affinity injection detaches semantic gradients from the affinity head;
- late-gated bias reaches only the configured final layers;
- delay and ramp progress are deterministic;
- structured affinity loss weights positive, hard-negative, and easy-negative
  groups independently;
- existing aliases still resolve to their original geometry switches and now
  select the intended instance-bias strategy;
- launcher parameters are optional and PowerShell syntax remains valid.

Only focused unit tests and compilation checks may run. Training and full data
conversion are explicitly excluded.
