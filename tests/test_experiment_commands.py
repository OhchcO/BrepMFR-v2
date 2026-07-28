import os
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMAND_FILE = PROJECT_ROOT / "tools" / "实验.txt"
EXPECTED_MODES = [
    "baseline",
    "topo_aag",
    "uv_aag",
    "pc_aag",
    "uv_pc_aag",
    "dynamic_no_loss",
    "affinity_only",
    "instance",
    "oracle",
    "cyl",
    "point",
    "spe_pcaag",
    "full",
    "cyl_affinity",
]


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


def test_experiment_file_has_one_copyable_train_and_test_block_per_mode():
    text = COMMAND_FILE.read_text(encoding="utf-8")
    actual_modes = re.findall(r"(?m)^\s+-Mode ([a-z_]+) `?$", text)

    assert actual_modes == EXPECTED_MODES
    assert text.count("run_mfrinstseg_test.ps1") == len(EXPECTED_MODES)
    assert text.count('Get-ChildItem -LiteralPath') == len(EXPECTED_MODES)
    assert "最优best模型的完整路径" not in text


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
