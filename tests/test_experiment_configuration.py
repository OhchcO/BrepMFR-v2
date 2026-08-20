import ast
import importlib
import json
import os
import re
import subprocess
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


def test_training_parser_defaults_to_repository_aag_assets(segmentation):
    args = segmentation.build_parser().parse_args(["train"])
    project_root = Path(segmentation.__file__).resolve().parent

    assert args.extractor_file == project_root / "tools" / "AAGExtractor.py"
    assert args.schema_file == project_root / "tools" / "schema.json"


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


def test_parametric_surface_cli_switches_are_configurable(segmentation):
    args = segmentation.build_parser().parse_args(
        [
            "train",
            "--dataset_path",
            "prepared",
            "--use_face_pointcloud",
            "--face_geometry_encoder",
            "param_surface",
            "--surface_neighbor_metric",
            "intrinsic",
            "--surface_trim_aware",
            "--surface_fusion",
            "gated",
        ]
    )
    segmentation._validate_configuration(segmentation.build_parser(), args)
    assert args.face_geometry_encoder == "param_surface"
    assert args.surface_neighbor_metric == "intrinsic"
    assert args.surface_trim_aware is True
    assert args.surface_fusion == "gated"


@pytest.mark.parametrize(
    (
        "alias",
        "mode",
        "uv",
        "point",
        "cyl",
        "bias_scale",
        "bias_strategy",
    ),
    [
        ("baseline", "baseline", True, False, False, 1.0, "legacy"),
        ("topo_aag", "baseline", False, False, False, 1.0, "legacy"),
        ("uv_aag", "baseline", True, False, False, 1.0, "legacy"),
        ("pc_aag", "baseline", False, True, False, 1.0, "legacy"),
        ("cyl_aag", "baseline", False, False, True, 1.0, "legacy"),
        ("uv_pc_aag", "baseline", True, True, False, 1.0, "legacy"),
        ("cyl", "baseline", True, False, True, 1.0, "legacy"),
        (
            "cyl_affinity",
            "dynamic_affinity",
            True,
            False,
            True,
            0.0,
            "legacy",
        ),
        ("point", "baseline", True, True, False, 1.0, "legacy"),
        ("spe_pcaag", "baseline", False, True, True, 1.0, "legacy"),
        (
            "spe_instance",
            "dynamic_affinity",
            False,
            True,
            True,
            1.0,
            "late_gated",
        ),
        (
            "dynamic_no_loss",
            "dynamic",
            True,
            False,
            False,
            1.0,
            "late_gated",
        ),
        (
            "pc_dynamic_no_loss",
            "dynamic",
            False,
            True,
            False,
            1.0,
            "late_gated",
        ),
        (
            "instance",
            "dynamic_affinity",
            True,
            False,
            False,
            1.0,
            "late_gated",
        ),
        (
            "pc_instance",
            "dynamic_affinity",
            False,
            True,
            False,
            1.0,
            "late_gated",
        ),
        (
            "affinity_only",
            "dynamic_affinity",
            True,
            False,
            False,
            0.0,
            "late_gated",
        ),
        (
            "pc_affinity_only",
            "dynamic_affinity",
            False,
            True,
            False,
            0.0,
            "late_gated",
        ),
        (
            "oracle",
            "oracle",
            True,
            False,
            False,
            1.0,
            "late_gated",
        ),
        (
            "pc_oracle",
            "oracle",
            False,
            True,
            False,
            1.0,
            "late_gated",
        ),
        (
            "full",
            "dynamic_affinity",
            True,
            True,
            True,
            1.0,
            "legacy",
        ),
    ],
)
def test_alias_mapping(
    segmentation,
    alias,
    mode,
    uv,
    point,
    cyl,
    bias_scale,
    bias_strategy,
):
    switches = segmentation.resolve_experiment_alias(alias)
    assert (
        switches.experiment_mode,
        switches.use_face_uv,
        switches.use_face_pointcloud,
        switches.use_cylindrical_pos,
        switches.instance_bias_scale,
        switches.instance_bias_strategy,
    ) == (mode, uv, point, cyl, bias_scale, bias_strategy)


def test_launcher_accepts_all_formal_experiment_modes():
    launcher = Path(__file__).resolve().parents[1] / "tools" / "run_mfrinstseg_experiments.ps1"
    powershell = """
$command = Get-Command '{0}'
$metadata = [System.Management.Automation.CommandMetadata]::new($command)
$attribute = $metadata.Parameters['Mode'].Attributes |
  Where-Object {{ $_ -is [System.Management.Automation.ValidateSetAttribute] }}
$attribute.ValidValues | ConvertTo-Json -Compress
""".format(str(launcher).replace("'", "''"))
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            powershell,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    accepted_modes = json.loads(completed.stdout)

    assert accepted_modes == [
        "baseline",
        "topo_aag",
        "uv_aag",
        "pc_aag",
        "cyl_aag",
        "uv_pc_aag",
        "dynamic_no_loss",
        "pc_dynamic_no_loss",
        "affinity_only",
        "pc_affinity_only",
        "instance",
        "pc_instance",
        "oracle",
        "pc_oracle",
        "cyl",
        "cyl_affinity",
        "point",
        "spe_pcaag",
        "spe_instance",
        "full",
    ]


def test_launcher_all_matrix_includes_param_surface_instance_aliases():
    launcher = Path(__file__).resolve().parents[1] / "tools" / "run_mfrinstseg_experiments.ps1"
    text = launcher.read_text(encoding="utf-8")
    all_matrix = re.search(
        r"if \(\$All\) \{\s*\$Modes = @\((.*?)\)\s*\}",
        text,
        flags=re.DOTALL,
    )

    assert all_matrix is not None
    modes = re.findall(r'"([a-z_]+)"', all_matrix.group(1))
    for alias in (
        "pc_dynamic_no_loss",
        "pc_affinity_only",
        "pc_instance",
        "pc_oracle",
    ):
        assert alias in modes


def test_training_parser_and_launcher_expose_custom_results_root(
    segmentation,
    tmp_path,
):
    args = segmentation.build_parser().parse_args(
        [
            "train",
            "--dataset_path",
            str(tmp_path),
            "--results_root",
            str(tmp_path / "custom-results"),
        ]
    )
    assert args.results_root == tmp_path / "custom-results"

    launcher = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run_mfrinstseg_experiments.ps1"
    )
    source = launcher.read_text(encoding="utf-8")
    assert "[string]$ResultBasePath" in source
    assert '"--results_root", $ResultBasePath' in source


def test_make_run_path_uses_custom_results_root(
    segmentation,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        segmentation.time,
        "strftime",
        lambda fmt: {"%m%d": "0730", "%H%M%S": "120000"}[fmt],
    )

    results_path, run_path, month_day, hour_min_second = (
        segmentation._make_run_path("experiment", tmp_path)
    )

    assert results_path == tmp_path / "experiment"
    assert run_path == tmp_path / "experiment" / "0730" / "120000"
    assert run_path.is_dir()
    assert (month_day, hour_min_second) == ("0730", "120000")


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
    assert args.use_face_uv is True
    assert args.random_rotate is False
    assert args.instance_bias_strategy == "legacy"
    assert args.instance_relation_mapping == "signed"
    assert args.instance_bias_last_layers == 2
    assert args.instance_bias_confidence_power == 1.0
    assert args.instance_bias_gate_init == 0.0
    assert args.instance_bias_max_gate_scale == 1.0
    assert args.instance_bias_delay_epochs == 10
    assert args.instance_bias_ramp_epochs == 10

    positive_args = segmentation.build_parser().parse_args(
        ["train", "--instance_relation_mapping", "positive_only"]
    )
    assert positive_args.instance_relation_mapping == "positive_only"

    no_uv_args = segmentation.build_parser().parse_args(["train", "--no_face_uv"])
    assert no_uv_args.use_face_uv is False


@pytest.mark.parametrize("gate_init", (0.0, -0.25))
def test_parser_accepts_zero_and_signed_initial_residual_gates(
    segmentation, gate_init
):
    parser = segmentation.build_parser()
    args = parser.parse_args(
        [
            "train",
            "--dataset_path",
            ".",
            "--instance_bias_gate_init",
            str(gate_init),
            "--instance_bias_max_gate_scale",
            "0.5",
        ]
    )

    segmentation._validate_configuration(parser, args)


@pytest.mark.parametrize("gate_init", (1.0, -1.0, 1.5, -1.5))
def test_parser_rejects_initial_gate_at_or_above_magnitude_limit(
    segmentation, capsys, gate_init
):
    parser = segmentation.build_parser()
    args = parser.parse_args(
        [
            "train",
            "--instance_bias_gate_init",
            str(gate_init),
            "--instance_bias_max_gate_scale",
            "1.0",
        ]
    )

    with pytest.raises(SystemExit):
        segmentation._validate_configuration(parser, args)

    assert (
        "--instance_bias_gate_init magnitude must be less than "
        "--instance_bias_max_gate_scale" in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    "option, value",
    [
        ("instance_bias_gate_init", float("nan")),
        ("instance_bias_gate_init", float("inf")),
        ("instance_bias_gate_init", float("-inf")),
        ("instance_bias_max_gate_scale", float("nan")),
        ("instance_bias_max_gate_scale", float("inf")),
        ("instance_bias_max_gate_scale", float("-inf")),
    ],
)
def test_parser_rejects_non_finite_residual_gate_configuration(
    segmentation, capsys, option, value
):
    parser = segmentation.build_parser()
    args = parser.parse_args(
        [
            "train",
            "--dataset_path",
            ".",
            "--{}={}".format(option, value),
        ]
    )

    with pytest.raises(SystemExit):
        segmentation._validate_configuration(parser, args)

    assert "--{} must be finite".format(option) in capsys.readouterr().err


def test_residual_instance_bias_constructor_defaults_to_zero_gate():
    from models.modules.instance_bias import InstanceRelationBias

    relation_bias = InstanceRelationBias(
        embedding_dim=4,
        num_heads=1,
        mode="dynamic",
        strategy="late_gated",
    )

    assert relation_bias.residual_gate(0).item() == 0.0


def test_encoder_and_model_fallback_defaults_keep_residual_gate_disabled():
    project_root = Path(__file__).resolve().parents[1]
    encoder_tree = ast.parse(
        (project_root / "models" / "modules" / "brep_encoder.py").read_text(
            encoding="utf-8"
        )
    )
    encoder = next(
        node
        for node in encoder_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BrepEncoder"
    )
    encoder_init = next(
        node
        for node in encoder.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    encoder_defaults = dict(
        zip(
            encoder_init.args.args[-len(encoder_init.args.defaults) :],
            encoder_init.args.defaults,
        )
    )
    gate_default = next(
        default
        for argument, default in encoder_defaults.items()
        if argument.arg == "instance_bias_gate_init"
    )
    assert ast.literal_eval(gate_default) == 0.0

    model_tree = ast.parse(
        (project_root / "models" / "brepseg_model.py").read_text(encoding="utf-8")
    )
    fallback_defaults = [
        node.args[2]
        for node in ast.walk(model_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) == 3
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "instance_bias_gate_init"
    ]
    assert [ast.literal_eval(default) for default in fallback_defaults] == [0.0]


def test_launcher_exposes_optional_instance_bias_controls():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run_mfrinstseg_experiments.ps1"
    )
    powershell = """
$command = Get-Command '{0}'
$parameters = [System.Management.Automation.CommandMetadata]::new($command).Parameters
@{{
  LastLayers = $parameters['InstanceBiasLastLayers'].ParameterType.FullName
  Confidence = $parameters['InstanceBiasConfidencePower'].ParameterType.FullName
  GateInit = $parameters['InstanceBiasGateInit'].ParameterType.FullName
  MaxGate = $parameters['InstanceBiasMaxGateScale'].ParameterType.FullName
  Delay = $parameters['InstanceBiasDelayEpochs'].ParameterType.FullName
  Ramp = $parameters['InstanceBiasRampEpochs'].ParameterType.FullName
  RelationMapping = $parameters['InstanceRelationMapping'].ParameterType.FullName
}} | ConvertTo-Json -Compress
""".format(str(launcher).replace("'", "''"))
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            powershell,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "LastLayers": "System.Int32",
        "Confidence": "System.Double",
        "GateInit": "System.Double",
        "MaxGate": "System.Double",
        "Delay": "System.Int32",
        "Ramp": "System.Int32",
        "RelationMapping": "System.String",
    }


def test_training_launcher_sets_deterministic_cublas_workspace(tmp_path):
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    for split in ("train", "val"):
        (dataset / (split + ".txt")).write_text(
            "sample-id\n",
            encoding="utf-8",
        )
    captured = tmp_path / "cublas-workspace.txt"
    fake_python = tmp_path / "fake-python.ps1"
    fake_python.write_text(
        """
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CapturedArguments
)
[System.IO.File]::WriteAllText(
    $env:MFRINSTSEG_CUBLAS_CAPTURE,
    $env:CUBLAS_WORKSPACE_CONFIG
)
exit 0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    launcher = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run_mfrinstseg_experiments.ps1"
    )
    quote = lambda value: str(value).replace("'", "''")
    command = (
        "& '{launcher}' -Mode baseline -AutoPrepare $false "
        "-DatasetPath '{dataset}' -PythonExe '{python}'"
    ).format(
        launcher=quote(launcher),
        dataset=quote(dataset),
        python=quote(fake_python),
    )
    environment = os.environ.copy()
    environment["MFRINSTSEG_CUBLAS_CAPTURE"] = str(captured)

    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert captured.read_text(encoding="utf-8") == ":4096:8"


def test_training_launcher_uses_repository_aag_assets_by_default(tmp_path):
    raw = tmp_path / "raw"
    (raw / "steps").mkdir(parents=True)
    (raw / "labels").mkdir()
    captured = tmp_path / "captured-args.txt"
    fake_python = tmp_path / "fake-python.ps1"
    fake_python.write_text(
        """
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CapturedArguments
)
[System.IO.File]::WriteAllLines(
    $env:MFRINSTSEG_TRAIN_ARGUMENTS,
    $CapturedArguments
)
exit 0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[1]
    launcher = project_root / "tools" / "run_mfrinstseg_experiments.ps1"
    command = (
        "& '{0}' -Mode baseline -RawDatasetPath '{1}' "
        "-PreparedDatasetPath '{2}' -PythonExe '{3}'"
    ).format(launcher, raw, tmp_path / "prepared", fake_python)
    environment = os.environ.copy()
    environment["MFRINSTSEG_TRAIN_ARGUMENTS"] = str(captured)

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    arguments = captured.read_text(encoding="utf-8").splitlines()
    extractor_index = arguments.index("--extractor_file")
    schema_index = arguments.index("--schema_file")
    assert Path(arguments[extractor_index + 1]) == project_root / "tools" / "AAGExtractor.py"
    assert Path(arguments[schema_index + 1]) == project_root / "tools" / "schema.json"


def test_launcher_forwards_positive_mapping_and_uses_positive_result_name(
    tmp_path,
):
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    for split in ("train", "val"):
        (dataset / (split + ".txt")).write_text(
            "sample-id\n",
            encoding="utf-8",
        )
    captured = tmp_path / "captured-args.txt"
    fake_python = tmp_path / "fake-python.ps1"
    fake_python.write_text(
        """
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CapturedArguments
)
[System.IO.File]::WriteAllLines(
    $env:MFRINSTSEG_TRAIN_ARGUMENTS,
    $CapturedArguments
)
exit 0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    launcher = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run_mfrinstseg_experiments.ps1"
    )
    quote = lambda value: str(value).replace("'", "''")
    command = (
        "& '{launcher}' -Mode oracle -AutoPrepare $false "
        "-DatasetPath '{dataset}' -PythonExe '{python}' "
        "-InstanceRelationMapping positive_only"
    ).format(
        launcher=quote(launcher),
        dataset=quote(dataset),
        python=quote(fake_python),
    )
    environment = os.environ.copy()
    environment["MFRINSTSEG_TRAIN_ARGUMENTS"] = str(captured)

    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    arguments = captured.read_text(encoding="utf-8").splitlines()
    mapping_index = arguments.index("--instance_relation_mapping")
    gate_init_index = arguments.index("--instance_bias_gate_init")
    name_index = arguments.index("--experiment_name")
    assert arguments[mapping_index + 1] == "positive_only"
    assert arguments[gate_init_index + 1] == "0"
    assert arguments[name_index + 1] == "MFRInstSeg_1pct_oracle_positive_seed2026"


@pytest.mark.parametrize(
    ("sample_percent", "expected_prefix"),
    [(0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct")],
)
def test_launcher_derives_result_identity_from_sample_percent(
    tmp_path,
    sample_percent,
    expected_prefix,
):
    dataset = tmp_path / "prepared"
    dataset.mkdir()
    for split in ("train", "val"):
        (dataset / (split + ".txt")).write_text(
            "sample-id\n",
            encoding="utf-8",
        )
    captured = tmp_path / "captured-args.txt"
    fake_python = tmp_path / "fake-python.ps1"
    fake_python.write_text(
        r'''
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CapturedArguments
)
[System.IO.File]::WriteAllLines(
    $env:MFRINSTSEG_TRAIN_ARGUMENTS,
    $CapturedArguments
)
exit 0
'''.strip()
        + "\n",
        encoding="utf-8",
    )
    launcher = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run_mfrinstseg_experiments.ps1"
    )
    quote = lambda value: str(value).replace("'", "''")
    command = (
        "& '{launcher}' -Mode baseline -AutoPrepare $false "
        "-DatasetPath '{dataset}' -PythonExe '{python}' "
        "-SamplePercent {sample_percent}"
    ).format(
        launcher=quote(launcher),
        dataset=quote(dataset),
        python=quote(fake_python),
        sample_percent=sample_percent,
    )
    environment = os.environ.copy()
    environment["MFRINSTSEG_TRAIN_ARGUMENTS"] = str(captured)

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    arguments = captured.read_text(encoding="utf-8").splitlines()
    name_index = arguments.index("--experiment_name")
    assert arguments[name_index + 1] == (
        f"MFRInstSeg_{expected_prefix}_baseline_seed2026"
    )


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
            bin_root=tmp_path / "shared",
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
            "--shared_bin_cache_path",
            str(tmp_path / "shared"),
        ]
    )

    root = segmentation.prepare_training_data(args)

    assert root == tmp_path / "prepared"
    assert captured["face_pointcloud"] is True
    assert captured["cylindrical_pos"] is True
    assert captured["sample_percent"] == 0.01
    assert captured["candidate_grids"] == (20, 32, 48, 64, 80)
    assert captured["shared_bin_cache_root"] == tmp_path / "shared"
    assert args.bin_root == tmp_path / "shared"
    assert args.preparation_reuse_status == "built"
    assert args.selected_split_counts == {"train": 1, "val": 1, "test": 1}


def test_powershell_launchers_propagate_shared_bin_cache_path():
    tools_root = Path(__file__).resolve().parents[1] / "tools"
    training = (tools_root / "run_mfrinstseg_experiments.ps1").read_text(
        encoding="utf-8"
    )
    evaluation = (tools_root / "run_mfrinstseg_test.ps1").read_text(
        encoding="utf-8"
    )

    assert "[string]$SharedBinCachePath" in training
    assert '"--shared_bin_cache_path", $SharedBinCachePath' in training
    assert "[string]$SharedBinCachePath" in evaluation
    assert '"--shared-bin-cache-path", $SharedBinCachePath' in evaluation


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
        ["--experiment_alias", "cyl_aag"],
        ["--experiment_alias", "cyl_affinity"],
        ["--experiment_alias", "spe_pcaag"],
        ["--experiment_alias", "spe_instance"],
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
        last_model_path = ""

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
        lambda experiment_name, results_root: (
            tmp_path,
            tmp_path / "run",
            "0726",
            "120000",
        ),
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
    assert all(options["bin_root"] == tmp_path for options in dataset_options)


def test_checkpoint_callbacks_save_metric_specific_best_and_single_last(
    segmentation, monkeypatch, tmp_path
):
    configurations = []

    class FakeCheckpoint:
        def __init__(self, **kwargs):
            configurations.append(kwargs)

    monkeypatch.setattr(segmentation, "ModelCheckpoint", FakeCheckpoint)

    callbacks = segmentation._build_checkpoint_callbacks(tmp_path)

    assert len(callbacks) == 2
    assert configurations == [
        {
            "monitor": "mean_iou",
            "mode": "max",
            "dirpath": str(tmp_path),
            "filename": "best-mIoU-{epoch:03d}-{mean_iou:.6f}",
            "save_top_k": 1,
            "save_last": True,
            "auto_insert_metric_name": False,
        },
        {
            "monitor": "macro_f1",
            "mode": "max",
            "dirpath": str(tmp_path),
            "filename": "best-Macro-F1-{epoch:03d}-{macro_f1:.6f}",
            "save_top_k": 1,
            "save_last": False,
            "auto_insert_metric_name": False,
        },
    ]


def test_checkpoint_log_records_primary_supplementary_and_last_paths(
    segmentation, tmp_path
):
    log_path = tmp_path / "log.txt"
    miou_checkpoint = SimpleNamespace(
        best_model_path="best-mIoU.ckpt",
        last_model_path="last.ckpt",
    )
    macro_f1_checkpoint = SimpleNamespace(
        best_model_path="best-Macro-F1.ckpt",
    )

    segmentation._append_checkpoint_paths(
        log_path,
        miou_checkpoint,
        macro_f1_checkpoint,
    )

    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "best_miou_checkpoint=best-mIoU.ckpt",
        "best_macro_f1_checkpoint=best-Macro-F1.ckpt",
        "last_checkpoint=last.ckpt",
    ]


def test_test_dataset_requirements_come_from_checkpoint_hparams(test_mfrinstseg):
    hparams = {
        "use_face_uv": False,
        "use_face_pointcloud": True,
        "use_cylindrical_pos": False,
    }

    requirements = test_mfrinstseg.geometry_requirements(hparams)

    assert requirements.use_face_uv is False
    assert requirements.require_face_pointcloud is True
    assert requirements.require_parametric_surface is False
    assert requirements.require_cylindrical_pos is False


def test_parametric_checkpoint_requires_surface_bin_fields(test_mfrinstseg):
    requirements = test_mfrinstseg.geometry_requirements(
        {
            "use_face_pointcloud": True,
            "face_geometry_encoder": "param_surface",
        }
    )
    assert requirements.require_face_pointcloud is True
    assert requirements.require_parametric_surface is True


def test_old_checkpoint_defaults_to_uv_enabled(test_mfrinstseg):
    requirements = test_mfrinstseg.geometry_requirements({})

    assert requirements.use_face_uv is True


def test_dynamic_full_test_does_not_require_instance_labels(test_mfrinstseg):
    assert test_mfrinstseg.requires_instance_for_test("dynamic_affinity") is False
    assert test_mfrinstseg.requires_instance_for_test("oracle") is True
    assert test_mfrinstseg.supports_instance_evaluation("dynamic") is True
    assert test_mfrinstseg.supports_instance_evaluation("dynamic_affinity") is True
    assert test_mfrinstseg.supports_instance_evaluation("baseline") is False


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
        "mean_class_accuracy": 1.0,
        "mean_iou": 1.0,
        "macro_f1": 1.0,
        "per_class": [],
        "confusion_matrix": [[2]],
        "seed": 2026,
        "use_face_uv": False,
        "use_face_pointcloud": True,
        "use_cylindrical_pos": True,
        "pointcloud_scale": 1.0,
        "cylindrical_scale": 1.0,
        "instance_bias_strategy": "late_gated",
        "instance_relation_mapping": "positive_only",
        "instance_bias_last_layers": 2,
        "instance_bias_confidence_power": 1.0,
        "instance_bias_gate_init": 0.05,
        "instance_bias_max_gate_scale": 1.0,
        "instance_bias_delay_epochs": 10,
        "instance_bias_ramp_epochs": 10,
        "checkpoint_instance_bias_scale": 1.0,
        "effective_instance_bias_scale": 0.0,
        "instance_bias_scale_overridden": True,
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
        "use_face_uv",
        "use_face_pointcloud",
        "use_cylindrical_pos",
        "pointcloud_scale",
        "cylindrical_scale",
        "instance_bias_strategy",
        "instance_relation_mapping",
        "instance_bias_last_layers",
        "instance_bias_confidence_power",
        "instance_bias_gate_init",
        "instance_bias_max_gate_scale",
        "instance_bias_delay_epochs",
        "instance_bias_ramp_epochs",
        "checkpoint_instance_bias_scale",
        "effective_instance_bias_scale",
        "instance_bias_scale_overridden",
        "prepared_dataset_path",
        "mean_class_accuracy",
    ):
        assert "{}={}".format(key, result[key]) in log


@pytest.mark.parametrize(
    "script_name",
    ("mfrinstseg_to_bin.py", "prepare_dataset.py"),
)
def test_direct_tool_scripts_bootstrap_project_root_before_project_imports(
    script_name,
):
    script = Path(__file__).parents[1] / "tools" / script_name
    tree = ast.parse(script.read_text(encoding="utf-8"))
    bootstrap_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "insert"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "path"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "sys"
    ]
    project_import_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.split(".", 1)[0] in ("tools", "utils")
    ]

    assert bootstrap_lines, "{} must bootstrap the project root".format(script)
    assert project_import_lines
    assert min(bootstrap_lines) < min(project_import_lines)
