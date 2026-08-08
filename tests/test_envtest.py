from pathlib import Path

import pytest

from envtest import (
    CheckResult,
    SmokeTestConfig,
    build_parser,
    build_model_args,
    exit_code,
    main,
    move_batch,
    resolve_sample,
    run_forward_smoke_test,
    select_device,
    validate_output,
)


class _BoolResult:
    def __init__(self, value):
        self._value = value

    def all(self):
        return self

    def item(self):
        return self._value


class _Output:
    def __init__(self, shape, finite=True):
        self.shape = shape
        self._finite = finite

    def isfinite(self):
        return _BoolResult(self._finite)


def _write_sample(root: Path, split: str, sample_id: str) -> Path:
    (root / (split + ".txt")).write_text(sample_id + "\n", encoding="utf-8")
    path = root / (sample_id + ".bin")
    path.touch()
    return path


@pytest.mark.parametrize(
    "requested,cuda_available,expected",
    [
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
        ("cpu", True, "cpu"),
        ("cuda", True, "cuda"),
    ],
)
def test_select_device_obeys_requested_policy(
    requested, cuda_available, expected
):
    assert select_device(requested, cuda_available) == expected


def test_select_device_rejects_unknown_value():
    with pytest.raises(ValueError, match="auto, cpu, or cuda"):
        select_device("tpu", cuda_available=False)


def test_explicit_cuda_fails_when_unavailable():
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        select_device("cuda", cuda_available=False)


def test_resolve_sample_uses_train_before_val(tmp_path):
    train_bin = _write_sample(tmp_path, "train", "train_part")
    _write_sample(tmp_path, "val", "val_part")

    assert resolve_sample(tmp_path, None) == (
        "train",
        "train_part",
        train_bin,
    )


def test_resolve_sample_falls_back_to_test(tmp_path):
    test_bin = _write_sample(tmp_path, "test", "test_part")

    assert resolve_sample(tmp_path, None) == (
        "test",
        "test_part",
        test_bin,
    )


def test_explicit_sample_must_belong_to_a_split(tmp_path):
    _write_sample(tmp_path, "train", "known")
    (tmp_path / "unknown.bin").touch()

    with pytest.raises(ValueError, match="not listed"):
        resolve_sample(tmp_path, "unknown")


def test_resolve_sample_rejects_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset directory"):
        resolve_sample(tmp_path / "missing", None)


def test_resolve_sample_rejects_dataset_without_splits(tmp_path):
    with pytest.raises(FileNotFoundError, match="No split manifest"):
        resolve_sample(tmp_path, None)


def test_resolve_sample_rejects_missing_bin(tmp_path):
    (tmp_path / "train.txt").write_text("missing\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="BIN file"):
        resolve_sample(tmp_path, None)


def test_resolve_sample_rejects_duplicate_bin_stems(tmp_path):
    (tmp_path / "train.txt").write_text("duplicate\n", encoding="utf-8")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "duplicate.bin").touch()
    (tmp_path / "b" / "duplicate.bin").touch()

    with pytest.raises(ValueError, match="Multiple BIN files"):
        resolve_sample(tmp_path, None)


def test_check_result_renders_hint_only_when_present():
    assert CheckResult("PASS", "python", "3.8.19").render() == (
        "[PASS] python: 3.8.19"
    )
    assert CheckResult("FAIL", "cuda", "unavailable", "use cpu").render() == (
        "[FAIL] cuda: unavailable | Hint: use cpu"
    )


def test_check_result_rejects_unknown_level():
    with pytest.raises(ValueError, match="PASS, WARN, or FAIL"):
        CheckResult("INFO", "python", "version")


def test_exit_code_is_nonzero_when_any_check_failed():
    assert exit_code([CheckResult("PASS", "python", "ok")]) == 0
    assert exit_code([CheckResult("WARN", "cuda", "using CPU")]) == 0
    assert exit_code([CheckResult("FAIL", "forward", "broken")]) == 1


def test_validate_output_accepts_finite_expected_shape():
    validate_output(_Output((7, 25)), face_count=7, num_classes=25)


def test_validate_output_rejects_wrong_shape():
    with pytest.raises(ValueError, match="expected output shape"):
        validate_output(_Output((7, 24)), face_count=7, num_classes=25)


def test_validate_output_rejects_non_finite_values():
    with pytest.raises(ValueError, match="non-finite"):
        validate_output(_Output((7, 25), finite=False), 7, 25)


def test_build_model_args_maps_baseline_inputs():
    args = build_model_args("baseline")

    assert args.experiment_mode == "baseline"
    assert args.use_face_uv is True
    assert args.use_face_pointcloud is False
    assert args.use_cylindrical_pos is False
    assert args.face_geometry_encoder == "mini_point"


def test_build_model_args_uses_formal_parameter_surface_for_pc_aag():
    args = build_model_args("pc_aag")

    assert args.experiment_mode == "baseline"
    assert args.use_face_uv is False
    assert args.use_face_pointcloud is True
    assert args.use_cylindrical_pos is False
    assert args.face_geometry_encoder == "param_surface"
    assert args.surface_neighbor_metric == "intrinsic"
    assert args.surface_trim_aware is False
    assert args.surface_fusion == "gated"


class _Movable:
    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


def test_move_batch_moves_tensor_like_values_and_preserves_metadata():
    tensor = _Movable()
    batch = {"tensor": tensor, "id": ("part",), "optional": None}

    moved = move_batch(batch, "cuda")

    assert moved == batch
    assert tensor.devices == ["cuda"]


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _FakeCuda:
    @staticmethod
    def is_available():
        return False


class _FakeTorch:
    __version__ = "test-torch"
    version = type("Version", (), {"cuda": None})()
    cuda = _FakeCuda()

    @staticmethod
    def no_grad():
        return _NoGrad()


class _FakeDataset:
    calls = []

    def __init__(self, **kwargs):
        self.calls.append(kwargs)
        self.file_paths = [Path(kwargs["root_dir"]) / "sample.bin"]

    def __getitem__(self, index):
        assert index == 0
        return "dataset-item"


class _FakeModel:
    calls = []

    def __init__(self, args):
        self.calls.append(("init", args.experiment_mode))

    def eval(self):
        self.calls.append(("eval",))
        return self

    def to(self, device):
        self.calls.append(("to", device))
        return self

    def __call__(self, batch):
        self.calls.append(("forward", tuple(batch["padding_mask"].shape)))
        return _Output((3, 25))


class _PaddingMask(_Movable):
    shape = (1, 3)

    def __invert__(self):
        return self

    def sum(self):
        return type("Count", (), {"item": lambda self: 3})()


def test_forward_pipeline_uses_real_order_through_injected_boundaries(tmp_path):
    _write_sample(tmp_path, "train", "sample")
    _FakeDataset.calls.clear()
    _FakeModel.calls.clear()
    collator_calls = []

    def fake_collator(items, multi_hop_max_dist, spatial_pos_max):
        collator_calls.append((items, multi_hop_max_dist, spatial_pos_max))
        return {"padding_mask": _PaddingMask()}

    components = {
        "torch": _FakeTorch(),
        "versions": {"DGL": "test-dgl", "NumPy": "test-numpy", "Lightning": "test-pl"},
        "dataset_class": _FakeDataset,
        "collator": fake_collator,
        "model_class": _FakeModel,
    }

    results = run_forward_smoke_test(
        SmokeTestConfig(tmp_path, "cpu", None, "baseline"),
        components=components,
    )

    assert not [result for result in results if result.level == "FAIL"]
    assert collator_calls == [(["dataset-item"], 16, 32)]
    assert _FakeModel.calls == [
        ("init", "baseline"),
        ("eval",),
        ("to", "cpu"),
        ("forward", (1, 3)),
    ]
    assert _FakeDataset.calls[0]["split"] == "train"


def test_cli_requires_dataset_path():
    parser = build_parser()

    with pytest.raises(SystemExit) as error:
        parser.parse_args([])

    assert error.value.code == 2


def test_cli_defaults_to_auto_baseline(tmp_path):
    args = build_parser().parse_args(["--dataset-path", str(tmp_path)])

    assert args.dataset_path == tmp_path
    assert args.device == "auto"
    assert args.experiment == "baseline"
    assert args.sample_id is None


def test_cli_preserves_explicit_options(tmp_path):
    args = build_parser().parse_args(
        [
            "--dataset-path",
            str(tmp_path),
            "--device",
            "cpu",
            "--experiment",
            "pc_aag",
            "--sample-id",
            "part-42",
        ]
    )

    assert args.device == "cpu"
    assert args.experiment == "pc_aag"
    assert args.sample_id == "part-42"


def test_cli_rejects_unknown_device(tmp_path):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(
            ["--dataset-path", str(tmp_path), "--device", "tpu"]
        )

    assert error.value.code == 2


def test_main_prints_ready_summary_for_success(tmp_path, capsys):
    received = []

    def runner(config):
        received.append(config)
        return [CheckResult("PASS", "Forward", "output shape (3, 25)")]

    code = main(
        ["--dataset-path", str(tmp_path), "--device", "cpu"],
        runner=runner,
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "[PASS] Forward: output shape (3, 25)" in output
    assert "ENVIRONMENT READY" in output
    assert received == [SmokeTestConfig(tmp_path, "cpu", None, "baseline")]


def test_main_prints_not_ready_summary_for_failure(tmp_path, capsys):
    def runner(config):
        return [
            CheckResult(
                "FAIL",
                "Dataset",
                "missing",
                "pass a prepared cache",
            )
        ]

    code = main(["--dataset-path", str(tmp_path)], runner=runner)

    output = capsys.readouterr().out
    assert code == 1
    assert "[FAIL] Dataset: missing | Hint: pass a prepared cache" in output
    assert "ENVIRONMENT NOT READY" in output
