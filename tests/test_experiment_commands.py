import os
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMAND_FILE = PROJECT_ROOT / "tools" / "实验.txt"
EXPECTED_EXPERIMENT_IDENTITIES = [
    ("baseline", "signed"),
    ("topo_aag", "signed"),
    ("uv_aag", "signed"),
    ("pc_aag", "signed"),
    ("uv_pc_aag", "signed"),
    ("dynamic_no_loss", "signed"),
    ("affinity_only", "signed"),
    ("instance", "signed"),
    ("oracle", "signed"),
    ("cyl", "signed"),
    ("point", "signed"),
    ("spe_pcaag", "signed"),
    ("full", "signed"),
    ("cyl_affinity", "signed"),
    ("dynamic_no_loss", "positive_only"),
    ("affinity_only", "positive_only"),
    ("instance", "positive_only"),
    ("oracle", "positive_only"),
]

EXPECTED_INSTANCE_EXPERIMENT_DESCRIPTIONS = {
    "dynamic_no_loss": "residual predicted bias without affinity supervision",
    "affinity_only": "true shared affinity supervision with residual bias disabled",
    "instance": "true shared affinity supervision plus residual predicted bias",
    "oracle": "GT relation through the same residual attention path",
}


def _wrapper_fixture(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "test.txt").write_text("sample-id\n", encoding="utf-8")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    captured = tmp_path / "captured-args.txt"
    fake_python = tmp_path / "fake-python.ps1"
    fake_python.write_text(
        """
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CapturedArguments
)
[System.IO.File]::WriteAllLines(
    $env:MFRINSTSEG_CAPTURED_ARGUMENTS,
    $CapturedArguments
)
exit 0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return prepared, checkpoint, fake_python, captured


def test_test_wrapper_forwards_explicit_zero_instance_bias_override(tmp_path):
    prepared, checkpoint, fake_python, captured = _wrapper_fixture(tmp_path)
    environment = os.environ.copy()
    environment["MFRINSTSEG_CAPTURED_ARGUMENTS"] = str(captured)

    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PROJECT_ROOT / "tools" / "run_mfrinstseg_test.ps1"),
            "-Checkpoint",
            str(checkpoint),
            "-PreparedDataset",
            str(prepared),
            "-PythonExe",
            str(fake_python),
            "-InstanceBiasScaleOverride",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    arguments = captured.read_text(encoding="utf-8").splitlines()
    option_index = arguments.index("--instance-bias-scale-override")
    assert arguments[option_index + 1] == "0"


def test_test_wrapper_rejects_negative_instance_bias_override(tmp_path):
    prepared, checkpoint, fake_python, captured = _wrapper_fixture(tmp_path)
    environment = os.environ.copy()
    environment["MFRINSTSEG_CAPTURED_ARGUMENTS"] = str(captured)

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PROJECT_ROOT / "tools" / "run_mfrinstseg_test.ps1"),
            "-Checkpoint",
            str(checkpoint),
            "-PreparedDataset",
            str(prepared),
            "-PythonExe",
            str(fake_python),
            "-InstanceBiasScaleOverride",
            "-0.1",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert not captured.exists()


def test_test_wrapper_sets_deterministic_cublas_workspace(tmp_path):
    prepared, checkpoint, fake_python, captured = _wrapper_fixture(tmp_path)
    cublas_capture = tmp_path / "cublas-workspace.txt"
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
    environment = os.environ.copy()
    environment["MFRINSTSEG_CUBLAS_CAPTURE"] = str(cublas_capture)

    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PROJECT_ROOT / "tools" / "run_mfrinstseg_test.ps1"),
            "-Checkpoint",
            str(checkpoint),
            "-PreparedDataset",
            str(prepared),
            "-PythonExe",
            str(fake_python),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert cublas_capture.read_text(encoding="utf-8") == ":4096:8"


def _experiment_identities(text):
    train_blocks = re.findall(
        r'(?ms)^powershell\.exe .*?run_mfrinstseg_experiments\.ps1" `\r?\n'
        r"(?P<arguments>.*?)(?=^\s*$)",
        text,
    )
    identities = []
    for arguments in train_blocks:
        mode = re.search(r"(?m)^\s+-Mode ([a-z_]+) `?$", arguments)
        relation_mapping = re.search(
            r"(?m)^\s+-InstanceRelationMapping ([a-z_]+) `?$", arguments
        )
        assert mode is not None
        identities.append(
            (mode.group(1), relation_mapping.group(1) if relation_mapping else "signed")
        )
    return identities


def test_experiment_file_has_expected_copyable_train_and_test_blocks():
    text = COMMAND_FILE.read_text(encoding="utf-8")

    assert _experiment_identities(text) == EXPECTED_EXPERIMENT_IDENTITIES
    assert text.count("run_mfrinstseg_test.ps1") == (
        len(EXPECTED_EXPERIMENT_IDENTITIES) + 3
    )
    assert text.count('Get-ChildItem -LiteralPath') == len(EXPECTED_EXPERIMENT_IDENTITIES)
    for mode, description in EXPECTED_INSTANCE_EXPERIMENT_DESCRIPTIONS.items():
        assert "# {}: {}".format(mode, description) in text
    assert "最优best模型的完整路径" not in text


def test_experiment_file_has_latest_bias_off_evaluations():
    text = COMMAND_FILE.read_text(encoding="utf-8")
    expected = [
        (
            "dynamic_no_loss_positive",
            r"D:\Code\BrepMFR\results\MFRInstSeg_1pct_dynamic_no_loss_positive_seed2026\0728\174518\best-080-0.165093.ckpt",
        ),
        (
            "instance_positive",
            r"D:\Code\BrepMFR\results\MFRInstSeg_1pct_instance_positive_seed2026\0728\164959\best-095-0.142707.ckpt",
        ),
        (
            "oracle_positive",
            r"D:\Code\BrepMFR\results\MFRInstSeg_1pct_oracle_positive_seed2026\0728\164959\best-086-0.152804.ckpt",
        ),
    ]

    for label, checkpoint in expected:
        marker = "# Bias OFF: " + label
        start = text.index(marker)
        next_marker = text.find("# Bias OFF:", start + len(marker))
        block = text[start : next_marker if next_marker != -1 else len(text)]
        assert '-Checkpoint "' + checkpoint + '" `' in block
        assert "-InstanceBiasScaleOverride 0" in block
        assert "-BatchSize 16" in block
        assert "-NumWorkers 0" in block


def test_experiment_file_is_valid_powershell_syntax():
    command_file = str(COMMAND_FILE).replace("'", "''")
    parser_command = """
$tokens = $null
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
  '{0}',
  [ref]$tokens,
  [ref]$errors
) | Out-Null
if ($errors.Count -ne 0) {{
  $errors | ForEach-Object {{ Write-Error $_.Message }}
  exit 1
}}
""".format(command_file)
    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            parser_command,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
