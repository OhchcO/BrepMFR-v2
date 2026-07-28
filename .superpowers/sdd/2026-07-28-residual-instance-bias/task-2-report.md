# Task 2 report: zero-initialized residual bias gates

## RED/GREEN evidence

- RED: `G:\ProgramData\anaconda3\envs\aagmspenv\python.exe -m pytest tests/test_instance_bias.py -k 'residual_gate' -q` produced 7 expected failures: `num_residual_gates` was unsupported and zero `gate_init` was rejected.
- GREEN (focused): the same command produced `9 passed, 12 deselected` after the minimal residual-gate implementation.
- GREEN (full target): `G:\ProgramData\anaconda3\envs\aagmspenv\python.exe -m pytest tests/test_instance_bias.py -q` produced `21 passed`.

## Files

- `models/modules/instance_bias.py`: replaces the single sigmoid gate with zero-initialized, signed, bounded per-layer residual gates; retains mean-absolute gate logging; removes gate multiplication from relation bias.
- `tests/test_instance_bias.py`: adds coverage for initialization, saturation, progress, gradients, independent layers, invalid offsets and constructor values; updates prior relation-bias and gradient-routing expectations.

## Commit

- `HEAD` — `feat: add zero initialized residual bias gates`

## Self-review

- `gate_init=0.0` maps to raw zero, yielding exact effective zero with derivative `max_gate_scale`.
- `tanh` bounds gates symmetrically, and progress is validated and applied only when a gate is requested.
- Late-gated relation bias remains scaled by `bias_scale` and `progress`, but no longer by a gate.
- No CLI defaults, encoder-layer attention, training, or dataset conversion were changed or run.

## Concerns

- None. Applying residual gates to attention-output differences is intentionally deferred to the follow-up task.
