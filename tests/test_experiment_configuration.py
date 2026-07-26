import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class _ParserTrainer:
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--max_epochs", type=int, default=1000)
        return parser


def _module(name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def segmentation(monkeypatch):
    """Import the entry point with lightweight training dependency doubles."""
    cudnn = SimpleNamespace(enabled=False, benchmark=False, deterministic=False)
    torch = _module("torch", backends=SimpleNamespace(cudnn=cudnn))
    pl = _module(
        "pytorch_lightning",
        Trainer=_ParserTrainer,
        seed_everything=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "pytorch_lightning", pl)
    monkeypatch.setitem(
        sys.modules,
        "pytorch_lightning.callbacks",
        _module("pytorch_lightning.callbacks", ModelCheckpoint=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "pytorch_lightning.loggers",
        _module("pytorch_lightning.loggers", TensorBoardLogger=object),
    )
    monkeypatch.setitem(sys.modules, "data.dataset", _module("data.dataset", CADSynth=object))
    monkeypatch.setitem(
        sys.modules, "models.brepseg_model", _module("models.brepseg_model", BrepSeg=object)
    )
    monkeypatch.setitem(
        sys.modules, "models.modules.utils.macro", _module("models.modules.utils.macro")
    )
    monkeypatch.setitem(
        sys.modules,
        "training_callbacks",
        _module("training_callbacks", TextMetricsLogger=object),
    )
    monkeypatch.setattr(sys, "argv", ["pytest", "--unexpected-pytest-option"])
    sys.modules.pop("segmentation", None)
    return importlib.import_module("segmentation")


@pytest.fixture
def test_mfrinstseg(monkeypatch):
    """Import the evaluator without loading dataset/model dependencies."""
    monkeypatch.setitem(sys.modules, "torch", _module("torch"))
    monkeypatch.setitem(
        sys.modules, "data.dataset", _module("data.dataset", CADSynth=object)
    )
    monkeypatch.setitem(
        sys.modules,
        "models.brepseg_model",
        _module("models.brepseg_model", BrepSeg=object),
    )
    sys.modules.pop("tools.test_mfrinstseg", None)
    return importlib.import_module("tools.test_mfrinstseg")


def test_importing_segmentation_does_not_parse_process_arguments(segmentation):
    assert callable(segmentation.build_parser)
    assert callable(segmentation.main)


@pytest.mark.parametrize(
    ("alias", "mode", "point", "cyl", "bias_scale"),
    [
        ("baseline", "baseline", False, False, 1.0),
        ("cyl", "baseline", False, True, 1.0),
        ("point", "baseline", True, False, 1.0),
        ("spe_pcaag", "baseline", True, True, 1.0),
        ("instance", "dynamic_affinity", False, False, 1.0),
        ("affinity_only", "dynamic_affinity", False, False, 0.0),
        ("full", "dynamic_affinity", True, True, 1.0),
    ],
)
def test_alias_mapping(segmentation, alias, mode, point, cyl, bias_scale):
    switches = segmentation.resolve_experiment_alias(alias)
    assert (
        switches.experiment_mode,
        switches.use_face_pointcloud,
        switches.use_cylindrical_pos,
        switches.instance_bias_scale,
    ) == (mode, point, cyl, bias_scale)


def test_parser_exposes_preparation_and_geometry_defaults(segmentation):
    args = segmentation.build_parser().parse_args(["train"])

    assert args.sample_percent == 0.01
    assert args.split_ratios == [0.8, 0.1, 0.1]
    assert args.point_sampling_mode == "area_adaptive"
    assert args.fixed_point_count == 64
    assert args.min_point_count == 32
    assert args.max_point_count == 96
    assert args.area_sampling_exponent == 0.5
    assert args.candidate_grids == [20, 32, 48, 64, 80]
    assert args.pointcloud_scale == 1.0
    assert args.cylindrical_scale == 1.0
    assert args.random_rotate is False


def test_automatic_preparation_always_requests_full_geometry_contract(
    segmentation, monkeypatch, tmp_path
):
    captured = {}

    class FakePreparationConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(segmentation, "PreparationConfig", FakePreparationConfig)
    monkeypatch.setattr(
        segmentation,
        "ensure_prepared_dataset",
        lambda config: SimpleNamespace(
            root=tmp_path / "prepared",
            train_ids=("train-1",),
            val_ids=("val-1",),
            test_ids=("test-1",),
            reused=False,
        ),
    )
    args = segmentation.build_parser().parse_args(
        [
            "train",
            "--auto_prepare",
            "--raw_dataset_path",
            str(tmp_path / "raw"),
            "--prepared_dataset_path",
            str(tmp_path / "prepared"),
            "--extractor_file",
            str(tmp_path / "extractor.py"),
            "--schema_file",
            str(tmp_path / "schema.json"),
        ]
    )

    root = segmentation.prepare_training_data(args)

    assert root == tmp_path / "prepared"
    assert captured["face_pointcloud"] is True
    assert captured["cylindrical_pos"] is True
    assert captured["sample_percent"] == 0.01
    assert captured["candidate_grids"] == (20, 32, 48, 64, 80)
    assert args.preparation_reuse_status == "built"
    assert args.selected_split_counts == {"train": 1, "val": 1, "test": 1}


def test_legacy_dataset_path_does_not_run_preparation(
    segmentation, monkeypatch, tmp_path
):
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    monkeypatch.setattr(
        segmentation,
        "ensure_prepared_dataset",
        lambda config: pytest.fail("legacy mode must not prepare data"),
    )
    args = segmentation.build_parser().parse_args(
        ["train", "--dataset_path", str(legacy_root)]
    )

    assert segmentation.prepare_training_data(args) == legacy_root
    assert args.preparation_reuse_status == "legacy"


@pytest.mark.parametrize(
    "geometry_args",
    [
        ["--use_cylindrical_pos"],
        ["--experiment_alias", "cyl"],
        ["--experiment_alias", "spe_pcaag"],
        ["--experiment_alias", "full"],
    ],
)
def test_random_rotation_is_rejected_with_cylindrical_positions(
    segmentation, geometry_args
):
    with pytest.raises(SystemExit):
        segmentation.main(["train", "--dataset_path", ".", "--random_rotate", *geometry_args])


def test_main_orders_seed_preparation_components_then_trainer(
    segmentation, monkeypatch, tmp_path
):
    events = []
    dataset_options = []

    class FakeDataset:
        def __init__(self, **kwargs):
            events.append("dataset")
            dataset_options.append(kwargs)

        def get_dataloader(self, **kwargs):
            events.append("loader")
            return object()

    class FakeModel:
        def __init__(self, args):
            events.append("model")

        def save_hyperparameters(self, configuration):
            events.append("hparams")

    class FakeTrainer:
        def __init__(self, **kwargs):
            events.append("trainer")

        def fit(self, model, train_loader, val_loader):
            events.append("fit")

    class FakeCheckpoint:
        best_model_path = ""

        def __init__(self, **kwargs):
            events.append("checkpoint")

    monkeypatch.setattr(
        segmentation,
        "seed_everything",
        lambda seed: events.append(("seed", seed)),
    )
    monkeypatch.setattr(
        segmentation,
        "prepare_training_data",
        lambda args: events.append("prepare") or tmp_path,
    )
    monkeypatch.setattr(
        segmentation,
        "_make_run_path",
        lambda experiment_name: (tmp_path, tmp_path / "run", "0726", "120000"),
    )
    monkeypatch.setattr(segmentation, "CADSynth", FakeDataset)
    monkeypatch.setattr(segmentation, "BrepSeg", FakeModel)
    monkeypatch.setattr(segmentation, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(
        segmentation,
        "TextMetricsLogger",
        lambda **kwargs: events.append("text_logger") or object(),
    )
    monkeypatch.setattr(
        segmentation,
        "TensorBoardLogger",
        lambda *args, **kwargs: events.append("tensorboard") or object(),
    )
    monkeypatch.setattr(segmentation.pl, "Trainer", FakeTrainer)

    assert (
        segmentation.main(
            [
                "train",
                "--dataset_path",
                str(tmp_path),
                "--experiment_alias",
                "spe_pcaag",
                "--num_workers",
                "0",
            ]
        )
        == 0
    )

    assert events[:2] == [("seed", 2026), "prepare"]
    assert events.index("trainer") > events.index("dataset")
    assert events.index("trainer") > events.index("model")
    assert events.index("trainer") > events.index("checkpoint")
    assert all(options["require_face_pointcloud"] for options in dataset_options)
    assert all(options["require_cylindrical_pos"] for options in dataset_options)
    assert all(options["random_rotate"] is False for options in dataset_options)


def test_test_dataset_requirements_come_from_checkpoint_hparams(test_mfrinstseg):
    hparams = {"use_face_pointcloud": True, "use_cylindrical_pos": False}

    requirements = test_mfrinstseg.geometry_requirements(hparams)

    assert requirements.require_face_pointcloud is True
    assert requirements.require_cylindrical_pos is False


def test_dynamic_full_test_does_not_require_instance_labels(test_mfrinstseg):
    assert test_mfrinstseg.requires_instance_for_test("dynamic_affinity") is False
    assert test_mfrinstseg.requires_instance_for_test("oracle") is True


def test_test_results_record_configuration_provenance(
    test_mfrinstseg, tmp_path
):
    result = {
        "checkpoint": "best.ckpt",
        "checkpoint_epoch": 3,
        "experiment_mode": "dynamic_affinity",
        "split": "test",
        "samples": 1,
        "faces": 2,
        "elapsed_seconds": 0.25,
        "test_loss": 0.5,
        "per_face_accuracy": 1.0,
        "mean_iou": 1.0,
        "macro_f1": 1.0,
        "per_class": [],
        "confusion_matrix": [[2]],
        "seed": 2026,
        "use_face_pointcloud": True,
        "use_cylindrical_pos": True,
        "pointcloud_scale": 1.0,
        "cylindrical_scale": 1.0,
        "prepared_dataset_path": "prepared",
        "require_instance": False,
        "batch_size": 1,
        "num_workers": 0,
        "device": "cpu",
    }

    log_path, json_path = test_mfrinstseg.write_results(tmp_path, result)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    log = log_path.read_text(encoding="utf-8")
    assert saved["seed"] == 2026
    for key in (
        "seed",
        "use_face_pointcloud",
        "use_cylindrical_pos",
        "pointcloud_scale",
        "cylindrical_scale",
        "prepared_dataset_path",
    ):
        assert "{}={}".format(key, result[key]) in log
