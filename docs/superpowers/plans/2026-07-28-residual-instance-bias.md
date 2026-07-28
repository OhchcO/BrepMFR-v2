# Residual Dynamic Instance Bias Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make affinity supervision update the shared B-Rep encoder and replace direct late-layer instance-bias injection with a zero-initialized residual biased-attention path.

**Architecture:** `InstanceRelationBias` predicts affinity from the live middle encoder state, but detaches the predicted relation before semantic injection only in `dynamic_affinity`. Each configured late `GraphEncoderLayer` computes its original self-attention and an instance-biased self-attention with shared parameters, then applies a bounded per-layer residual gate initialized to zero.

**Tech Stack:** Python, PyTorch, fairseq attention layers, pytest, PowerShell launchers.

## Global Constraints

- Do not change the raw dataset, BIN schema, geometry encoders, semantic head, or dataset preparation.
- Preserve the user-facing aliases `baseline`, `dynamic_no_loss`, `affinity_only`, `instance`, and `oracle`.
- Use `positive_only` as the recommended relation mapping without removing `signed`.
- New residual gates default to exactly zero.
- Do not run training or full dataset conversion.
- Verification is limited to focused unit tests, compilation, and lightweight synthetic forward/backward checks.

---

## File Structure

- `models/modules/instance_bias.py`: affinity prediction, relation-gradient routing, bounded per-layer residual gates, and instance-bias construction.
- `models/modules/layers/brep_encoder_layer.py`: shared-parameter base/biased attention computation and residual mixing.
- `models/modules/brep_encoder.py`: middle-layer prediction boundary and final-layer residual-bias routing.
- `models/brepseg_model.py`: unchanged training loss contract; reads the updated effective gate for logging.
- `segmentation.py`: zero gate default and validation.
- `tools/run_mfrinstseg_experiments.ps1`: zero gate launcher default.
- `tests/test_instance_bias.py`: gradient-routing and gate tests.
- `tests/test_residual_instance_attention.py`: isolated layer-level residual attention tests.
- `tests/test_experiment_configuration.py`: CLI and alias regression tests.

### Task 1: Correct affinity gradient routing

**Files:**
- Modify: `tests/test_instance_bias.py`
- Modify: `models/modules/instance_bias.py`

**Interfaces:**
- Consumes: `InstanceRelationBias.forward(...) -> tuple[bias, logits]`.
- Produces: dynamic-affinity logits connected to `node_features`; injected dynamic-affinity bias detached from the affinity predictor; dynamic bias remains connected to its predictor.

- [ ] **Step 1: Replace the old detached-feature assertion with failing gradient tests**

Add tests that use `node_features = relation_inputs()["node_features"].clone().requires_grad_(True)`:

```python
def test_dynamic_affinity_loss_updates_input_representation():
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=2,
        mode="dynamic_affinity",
        affinity_dim=4,
        strategy="late_gated",
        bias_scale=1.0,
        gate_init=0.0,
    )
    inputs = relation_inputs()
    inputs["node_features"] = inputs["node_features"].clone().requires_grad_(True)

    _, logits = module(**inputs, progress=1.0)
    logits.sum().backward()

    assert inputs["node_features"].grad is not None
    assert torch.count_nonzero(inputs["node_features"].grad).item() > 0
```

Keep and update the existing semantic-bias test so `bias.sum().backward()`
does not produce gradients on `node_projection`, `spatial_projection`,
`d2_projection`, or `angle_projection` in `dynamic_affinity`. Keep the
`dynamic` test proving the same bias path can update its predictor.

- [ ] **Step 2: Run the new gradient test and verify RED**

Run:

```powershell
python -m pytest tests/test_instance_bias.py::test_dynamic_affinity_loss_updates_input_representation -q
```

Expected: FAIL because `_dynamic_logits` currently receives
`node_features.detach()`.

- [ ] **Step 3: Implement prediction-guidance decoupling**

In `InstanceRelationBias.forward`, always pass live `node_features` into
`_dynamic_logits`:

```python
logits = self._dynamic_logits(
    node_features,
    padding_mask,
    spatial_pos,
    d2_distance,
    angle_distance,
)
```

Build the relation normally, then detach only the relation used as semantic
bias for `dynamic_affinity`:

```python
relation = confidence_weighted_relation(
    logits,
    power=self.confidence_power,
    relation_mapping=self.relation_mapping,
)
if self.mode == "dynamic_affinity":
    relation = relation.detach()
```

- [ ] **Step 4: Run focused gradient tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_instance_bias.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add tests/test_instance_bias.py models/modules/instance_bias.py
git commit -m "fix: share affinity supervision with encoder"
```

### Task 2: Add zero-initialized bounded residual gates

**Files:**
- Modify: `tests/test_instance_bias.py`
- Modify: `models/modules/instance_bias.py`

**Interfaces:**
- Consumes: `gate_init`, `max_gate_scale`, and configured late-layer count.
- Produces: `InstanceRelationBias.residual_gate(layer_offset, progress) -> Tensor`; `gate_scale` remains available as the mean effective gate for logging.

- [ ] **Step 1: Write failing gate tests**

Add:

```python
def test_residual_gates_start_at_exactly_zero_and_are_bounded():
    module = InstanceRelationBias(
        embedding_dim=4,
        num_heads=2,
        mode="dynamic_affinity",
        affinity_dim=4,
        strategy="late_gated",
        gate_init=0.0,
        max_gate_scale=0.75,
        num_residual_gates=2,
    )

    assert module.residual_gate(0, progress=1.0).item() == 0.0
    assert module.residual_gate(1, progress=1.0).item() == 0.0

    with torch.no_grad():
        module.gate_logits.fill_(100.0)
    assert torch.all(module.residual_gates(progress=1.0) <= 0.75)
```

Add validation tests for an invalid layer offset and for `gate_init` outside
`[-max_gate_scale, max_gate_scale]`.

- [ ] **Step 2: Run the gate test and verify RED**

Run:

```powershell
python -m pytest tests/test_instance_bias.py::test_residual_gates_start_at_exactly_zero_and_are_bounded -q
```

Expected: FAIL because per-layer residual gates do not exist and the current
sigmoid gate cannot represent exact zero.

- [ ] **Step 3: Implement per-layer signed bounded gates**

Add `num_residual_gates=1` to `InstanceRelationBias.__init__`. For non-baseline
late-gated modes, initialize a parameter vector using inverse `tanh`:

```python
gate_fraction = float(gate_init) / self.max_gate_scale
gate_fraction = max(min(gate_fraction, 1.0 - 1e-6), -1.0 + 1e-6)
self.gate_logits = nn.Parameter(
    torch.full(
        (int(num_residual_gates),),
        torch.atanh(torch.tensor(gate_fraction)).item(),
    )
)
```

Expose:

```python
def residual_gates(self, progress=1.0):
    if self.strategy != "late_gated" or self.mode == "baseline":
        return None
    return (
        self.max_gate_scale
        * torch.tanh(self.gate_logits)
        * float(progress)
    )

def residual_gate(self, layer_offset, progress=1.0):
    gates = self.residual_gates(progress)
    if gates is None or not 0 <= layer_offset < gates.numel():
        raise IndexError("residual gate layer offset is out of range")
    return gates[layer_offset]
```

Retain `gate_scale` as the mean absolute configured gate for existing logging.
Remove gate multiplication from `face_bias`; `bias_scale` and `progress`
continue to scale the relation bias, while the new gate is applied to the
attention-output difference.

- [ ] **Step 4: Run the module tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_instance_bias.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add tests/test_instance_bias.py models/modules/instance_bias.py
git commit -m "feat: add zero initialized residual bias gates"
```

### Task 3: Implement residual biased attention inside one encoder layer

**Files:**
- Create: `tests/test_residual_instance_attention.py`
- Modify: `models/modules/layers/brep_encoder_layer.py`

**Interfaces:**
- Consumes: optional `instance_attn_bias: Tensor` and optional scalar `instance_attn_gate: Tensor`.
- Produces: the existing `GraphEncoderLayer.forward(...) -> tuple[x, attn]` contract.

- [ ] **Step 1: Write failing layer equivalence and activation tests**

Create a small deterministic layer with all dropout values set to zero. Deep
copy it to compare the old base call with the new optional arguments:

```python
def test_zero_gate_is_exactly_baseline_attention():
    torch.manual_seed(2026)
    layer = GraphEncoderLayer(
        embedding_dim=8,
        ffn_embedding_dim=16,
        num_attention_heads=2,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    ).eval()
    x = torch.randn(4, 1, 8)
    graph_bias = torch.zeros(1, 2, 4, 4)
    instance_bias = torch.zeros_like(graph_bias)
    instance_bias[:, :, 1, 2] = 3.0

    expected, _ = layer(x, self_attn_bias=graph_bias)
    actual, _ = layer(
        x,
        self_attn_bias=graph_bias,
        instance_attn_bias=instance_bias,
        instance_attn_gate=torch.tensor(0.0),
    )

    assert torch.equal(actual, expected)
```

Add a second test with gate `1.0` asserting the result differs. Add a backward
test asserting the zero-initialized gate receives a nonzero gradient when the
biased and base attention outputs differ.

- [ ] **Step 2: Run the new layer tests and verify RED**

Run:

```powershell
python -m pytest tests/test_residual_instance_attention.py -q
```

Expected: FAIL because `GraphEncoderLayer.forward` does not accept the new
arguments.

- [ ] **Step 3: Implement shared-parameter dual attention**

Extend `GraphEncoderLayer.forward` with:

```python
instance_attn_bias: Optional[torch.Tensor] = None,
instance_attn_gate: Optional[torch.Tensor] = None,
```

Compute the base attention exactly as before. When both optional values are
present, call the same `self.self_attn` module using
`self_attn_bias + instance_attn_bias`, then mix before the existing dropout and
outer residual:

```python
base_x, attn = self.self_attn(...)
if instance_attn_bias is not None and instance_attn_gate is not None:
    biased_x, _ = self.self_attn(
        query=attention_input,
        key=attention_input,
        value=attention_input,
        attn_bias=self_attn_bias + instance_attn_bias,
        key_padding_mask=self_attn_padding_mask,
        need_weights=False,
        attn_mask=self_attn_mask,
    )
    x = base_x + instance_attn_gate * (biased_x - base_x)
else:
    x = base_x
```

Keep `attention_input` separate from the later output variable so pre-layer
normalization works identically in both calls. Wrap only the optional biased
call in a torch RNG-state preservation context so an exactly zero gate does not
advance the random stream seen by the remaining layer operations.

- [ ] **Step 4: Run the layer tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_residual_instance_attention.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add tests/test_residual_instance_attention.py models/modules/layers/brep_encoder_layer.py
git commit -m "feat: mix residual instance biased attention"
```

### Task 4: Route residual bias through only the configured final layers

**Files:**
- Modify: `tests/test_instance_bias.py`
- Modify: `models/modules/brep_encoder.py`

**Interfaces:**
- Consumes: `InstanceRelationBias.residual_gate(layer_offset, progress)`.
- Produces: ordinary `self_attn_bias` for every layer plus optional
  `instance_attn_bias` and `instance_attn_gate` only for final configured
  layers.

- [ ] **Step 1: Write a failing encoder-routing test**

Build a four-layer lightweight `BrepEncoder`, replace its layers with recording
test doubles that preserve the call signature, and assert:

```python
assert calls[0]["instance_attn_bias"] is None
assert calls[1]["instance_attn_bias"] is None
assert calls[2]["instance_attn_bias"] is not None
assert calls[3]["instance_attn_bias"] is not None
assert calls[2]["self_attn_bias"] is graph_bias
```

The test must also assert that the base graph bias never already contains the
instance bias.

- [ ] **Step 2: Run the routing test and verify RED**

Run:

```powershell
python -m pytest tests/test_instance_bias.py -k residual_bias_reaches_only_final_layers -q
```

Expected: FAIL because `BrepEncoder` currently adds instance bias directly to
`layer_attn_bias`.

- [ ] **Step 3: Implement late-layer residual routing**

Construct `InstanceRelationBias` with:

```python
num_residual_gates=self.instance_bias_last_layers
```

Replace:

```python
layer_attn_bias = attn_bias + instance_bias
```

with independent arguments:

```python
instance_attn_bias = None
instance_attn_gate = None
if use_late_gated_bias and layer_index >= late_layer_start:
    layer_offset = layer_index - late_layer_start
    instance_attn_bias = instance_bias
    instance_attn_gate = self.instance_relation_bias.residual_gate(
        layer_offset,
        progress=instance_bias_progress,
    )

x, _ = layer(
    x,
    self_attn_padding_mask=padding_mask,
    self_attn_mask=attn_mask,
    self_attn_bias=attn_bias,
    instance_attn_bias=instance_attn_bias,
    instance_attn_gate=instance_attn_gate,
)
```

Legacy strategy continues to add its instance bias directly before the layer
loop.

- [ ] **Step 4: Run module and routing tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_instance_bias.py tests/test_residual_instance_attention.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```powershell
git add tests/test_instance_bias.py models/modules/brep_encoder.py
git commit -m "feat: route residual bias through late layers"
```

### Task 5: Update zero-gate configuration and experiment regressions

**Files:**
- Modify: `tests/test_experiment_configuration.py`
- Modify: `tests/test_experiment_commands.py`
- Modify: `segmentation.py`
- Modify: `tools/run_mfrinstseg_experiments.ps1`
- Modify: `models/brepseg_model.py` only if the gate logging accessor requires adaptation.

**Interfaces:**
- Consumes: existing `--instance_bias_gate_init` and PowerShell
  `-InstanceBiasGateInit`.
- Produces: default `0.0`; accepts values in
  `[-instance_bias_max_gate_scale, instance_bias_max_gate_scale]`.

- [ ] **Step 1: Write failing configuration tests**

Change parser and launcher expectations from `0.05` to `0.0`. Add parser
validation coverage proving zero is accepted and a magnitude greater than or
equal to the maximum is rejected. Keep all alias mapping expectations
unchanged.

- [ ] **Step 2: Run configuration tests and verify RED**

Run:

```powershell
python -m pytest tests/test_experiment_configuration.py tests/test_experiment_commands.py -q
```

Expected: FAIL because Python and PowerShell defaults are still `0.05`, and
validation rejects zero.

- [ ] **Step 3: Update defaults and validation**

In `segmentation.py`, use:

```python
parser.add_argument("--instance_bias_gate_init", type=float, default=0.0)
```

Validate:

```python
if abs(args.instance_bias_gate_init) >= args.instance_bias_max_gate_scale:
    parser.error(
        "--instance_bias_gate_init magnitude must be smaller than "
        "--instance_bias_max_gate_scale"
    )
```

Update the PowerShell launcher parameter default to `0.0`. Preserve parameter
names and command construction. Adapt gate logging to report the mean absolute
effective residual gate without changing the log key
`train_instance_bias_gate`.

- [ ] **Step 4: Run configuration tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_experiment_configuration.py tests/test_experiment_commands.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

```powershell
git add tests/test_experiment_configuration.py tests/test_experiment_commands.py segmentation.py tools/run_mfrinstseg_experiments.ps1 models/brepseg_model.py
git commit -m "chore: default residual instance gate to zero"
```

### Task 6: Focused integration verification

**Files:**
- Modify: `tools/实验.txt` only if its comments incorrectly describe the new
  gradient or residual behavior.

**Interfaces:**
- Consumes: all preceding implementation changes.
- Produces: verified lightweight training/evaluation entry points and updated
  copy-ready experiment descriptions.

- [ ] **Step 1: Update experiment comments without changing copy-ready commands**

Describe:

```text
dynamic_no_loss: residual predicted bias without affinity supervision
affinity_only: true shared affinity supervision with residual gate disabled
instance: true shared affinity supervision plus residual predicted bias
oracle: GT relation through the same residual attention path
```

- [ ] **Step 2: Run focused tests**

Run:

```powershell
python -m pytest tests/test_instance_bias.py tests/test_residual_instance_attention.py tests/test_affinity_training.py tests/test_experiment_configuration.py tests/test_experiment_commands.py -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 3: Compile modified Python files**

Run:

```powershell
python -m py_compile models/modules/instance_bias.py models/modules/layers/brep_encoder_layer.py models/modules/brep_encoder.py models/brepseg_model.py segmentation.py
```

Expected: exit code 0 and no output.

- [ ] **Step 4: Review the final diff**

Run:

```powershell
git diff --check
git status --short
git diff --stat
```

Expected: no whitespace errors; only the planned model, configuration, test,
and experiment-description files are changed.

- [ ] **Step 5: Commit final documentation adjustment**

```powershell
git add tools/实验.txt
git commit -m "docs: update residual instance experiment commands"
```

