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
