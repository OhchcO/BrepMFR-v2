# Environment Smoke-Test Design

## Objective

Add a read-only `envtest.py` entry point that determines whether the current
Python environment, prepared MFRInstSeg dataset, and selected compute device
can execute the project's real data-loading and model-forward path.

The smoke test must not train, convert data, modify manifests, or write model
checkpoints.

## Command-Line Interface

The script is launched from the repository root:

```powershell
python envtest.py `
  --dataset-path "D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_10pct_seed42" `
  --device auto
```

Supported arguments:

- `--dataset-path`: required prepared BIN dataset directory.
- `--device`: `auto`, `cpu`, or `cuda`; defaults to `auto`.
- `--sample-id`: optional sample ID. When omitted, use the first usable entry
  from `train.txt`, then `val.txt`, then `test.txt`.
- `--experiment`: experiment alias used to configure required fields; defaults
  to `baseline` so the smoke test can run against both legacy and full-geometry
  caches.

`auto` selects CUDA when PyTorch reports it available, otherwise CPU. An
explicit `cuda` request fails when CUDA is unavailable.

## Checks and Data Flow

The script performs these stages in order:

1. Report the Python executable and versions of Python, PyTorch, DGL, NumPy,
   PyTorch Lightning, and CUDA-related runtime information.
2. Resolve the requested device and report the CUDA device name and memory
   when applicable.
3. Validate the prepared dataset directory and locate `train.txt`, `val.txt`,
   or `test.txt`.
4. Resolve one sample ID and verify that its BIN file exists.
5. Load the sample through the project's existing dataset and collator path,
   not through an independent BIN parser.
6. Resolve the selected experiment alias and construct the existing
   segmentation model with lightweight dimensions suitable for a smoke test.
7. Move the batch and model to the selected device and execute one
   `torch.no_grad()` forward pass.
8. Verify that the semantic output has one row per face, has the configured
   class dimension, and contains only finite values.

The experiment alias controls whether UV, point-cloud, cylindrical, or
instance fields are required. This makes the script useful both for a minimal
baseline check and for validating the exact `pc_aag` environment needed by the
10% matrix.

## Reporting and Exit Status

Each stage emits one of:

- `[PASS]`: the check succeeded;
- `[WARN]`: optional acceleration or metadata is unavailable, but the selected
  smoke test remains executable;
- `[FAIL]`: the requested forward path cannot run.

Failures include the failing stage, exception type, message, and a concise
remediation hint. The script returns exit code `0` only when the sample forward
pass and output validation succeed; otherwise it returns a nonzero exit code.

Expected environment failures are caught and reported without a Python
traceback by default. Unexpected failures retain enough exception context to
diagnose the problem.

## Isolation and Reuse

Environment-independent behavior is kept in small functions for:

- device selection;
- split/sample resolution;
- structured check-result formatting;
- forward-output validation.

Project-specific model and dataset imports occur only when their check stage
runs, allowing missing dependencies to be reported cleanly rather than causing
an immediate import-time crash.

No existing training or model behavior is changed.

## Testing

Unit tests cover:

- `auto`, explicit CPU, and unavailable explicit CUDA device selection;
- split priority and explicit sample selection;
- missing dataset, split, and BIN error reporting;
- PASS/WARN/FAIL formatting and aggregate exit status;
- rejection of non-finite or incorrectly shaped forward outputs.

Tests use temporary prepared-dataset fixtures and lightweight doubles at the
dependency boundary. A manual invocation against a real BIN cache is the final
integration verification when the required Python environment is available.

