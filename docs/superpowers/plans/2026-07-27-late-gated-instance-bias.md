# Late-Gated Dynamic Instance Bias Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace new alias-launched instance experiments with a confidence-aware, late-layer gated dynamic bias while retaining the original BrepMFR baseline and existing experiment switches.

**Architecture:** Preserve a legacy instance-bias strategy for old checkpoints and add a `late_gated` strategy for the existing dynamic aliases. Predict affinity from a detached middle representation, inject a signed confidence-weighted relation only into final encoder layers, and train the affinity head with deterministic positive/hard-negative/easy-negative group balancing.

**Tech Stack:** Python, PyTorch, PyTorch Lightning, pytest, PowerShell.

## Global Constraints

- Do not change dataset conversion, BIN schema, semantic classifier, point-cloud branch, cylindrical branch, or original baseline behavior.
- Keep `baseline`, `dynamic_no_loss`, `affinity_only`, `instance`, and `oracle` as the user-facing switches.
- Old checkpoints without the new strategy hyperparameter must retain legacy construction.
- Do not run model training or full-dataset conversion.
- Use test-first red-green cycles for every production behavior change.

---

### Task 1: Signed confidence-aware relation module

**Files:**
- Modify: `tests/test_instance_bias.py`
- Modify: `models/modules/instance_bias.py`

**Interfaces:**
- Consumes: node embeddings and existing pair descriptors
- Produces: `InstanceRelationBias.forward(..., progress: float) -> (bias, logits)`

- [ ] **Step 1: Write failing tests**

Test that zero logits produce zero late-gated relation, positive/negative logits
produce positive/negative bias, invalid progress is rejected, scale zero retains
logits but returns zero bias, and dynamic-affinity relation injection does not
send semantic gradients into affinity-head parameters.

- [ ] **Step 2: Run the focused test and verify expected failures**

```powershell
pytest tests/test_instance_bias.py -q
```

- [ ] **Step 3: Implement the minimal late-gated relation path**

Add strategy validation, learnable bounded gate, signed confidence transform,
progress scaling, optional relation detachment, and legacy-compatible behavior.

- [ ] **Step 4: Run the focused test**

```powershell
pytest tests/test_instance_bias.py -q
```

### Task 2: Mid-encoder prediction and late-layer injection

**Files:**
- Modify: `tests/test_geometry_encoders.py`
- Modify: `models/modules/brep_encoder.py`

**Interfaces:**
- Consumes: `instance_bias_last_layers`, `instance_bias_progress`
- Produces: early layers using graph bias only and final layers using graph plus instance bias

- [ ] **Step 1: Write failing recording-layer tests**

Record the attention bias received by every encoder layer and assert that only
the configured final layers receive instance bias. Assert that the relation
head receives the representation after the middle boundary and is called once.

- [ ] **Step 2: Run the focused test and verify expected failures**

```powershell
pytest tests/test_geometry_encoders.py -q
```

- [ ] **Step 3: Implement the minimal encoder scheduling**

Keep the legacy all-layer path unchanged. For `late_gated`, predict once at the
middle boundary and add the resulting bias only to final executed layers.

- [ ] **Step 4: Run the focused test**

```powershell
pytest tests/test_geometry_encoders.py -q
```

### Task 3: Structured affinity loss and epoch schedule

**Files:**
- Create: `tests/test_affinity_training.py`
- Modify: `models/brepseg_model.py`

**Interfaces:**
- Consumes: flattened semantic labels, padded instance targets, padding mask
- Produces: deterministic three-group affinity loss and bias progress in `[0, 1]`

- [ ] **Step 1: Write failing tests**

Use hand-computed logits to verify independent positive, hard-negative, and
easy-negative group means. Verify delay/ramp values before, during, and after
the ramp.

- [ ] **Step 2: Run the focused test and verify expected failures**

```powershell
pytest tests/test_affinity_training.py -q
```

- [ ] **Step 3: Implement structured loss and schedule**

Extend `balanced_affinity_loss` with semantic labels and add a pure helper for
epoch progress. Pass progress through training and validation encoder calls.

- [ ] **Step 4: Run the focused test**

```powershell
pytest tests/test_affinity_training.py -q
```

### Task 4: Existing alias and launcher compatibility

**Files:**
- Modify: `tests/test_experiment_configuration.py`
- Modify: `tests/test_experiment_commands.py`
- Modify: `segmentation.py`
- Modify: `tools/run_mfrinstseg_experiments.ps1`

**Interfaces:**
- Consumes: original experiment aliases and optional PowerShell parameters
- Produces: new strategy hyperparameters threaded into `BrepSeg` without changing command shape

- [ ] **Step 1: Write failing parser and alias tests**

Assert that instance aliases resolve to `late_gated`, baseline remains legacy
and geometry switches remain byte-for-byte identical. Assert parser defaults
and launcher parameters for last layers, gate initialization, confidence power,
delay, and ramp.

- [ ] **Step 2: Run the focused tests and verify expected failures**

```powershell
pytest tests/test_experiment_configuration.py tests/test_experiment_commands.py -q
```

- [ ] **Step 3: Thread configuration with minimal edits**

Extend `ExperimentSwitches`, parser validation, model construction, and the
PowerShell argument array. Do not add or rename experiment modes.

- [ ] **Step 4: Run the focused tests**

```powershell
pytest tests/test_experiment_configuration.py tests/test_experiment_commands.py -q
```

### Task 5: Regression verification

**Files:**
- Verify all files changed by Tasks 1-4

**Interfaces:**
- Consumes: completed implementation
- Produces: fresh evidence without training

- [ ] **Step 1: Run focused instance/configuration tests**

```powershell
pytest tests/test_instance_bias.py tests/test_affinity_training.py tests/test_geometry_encoders.py tests/test_experiment_configuration.py tests/test_experiment_commands.py -q
```

- [ ] **Step 2: Run the repository unit-test suite**

```powershell
pytest tests -q
```

- [ ] **Step 3: Compile changed Python modules**

```powershell
python -m py_compile segmentation.py models/brepseg_model.py models/modules/brep_encoder.py models/modules/instance_bias.py
```

- [ ] **Step 4: Parse the PowerShell launcher**

```powershell
powershell.exe -NoProfile -Command "[scriptblock]::Create((Get-Content -LiteralPath 'tools/run_mfrinstseg_experiments.ps1' -Raw)) | Out-Null"
```

- [ ] **Step 5: Review the exact diff**

```powershell
git diff --check
git diff -- models/modules/instance_bias.py models/modules/brep_encoder.py models/brepseg_model.py segmentation.py tools/run_mfrinstseg_experiments.ps1 tests
```
