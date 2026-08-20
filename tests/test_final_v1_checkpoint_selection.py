import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "tools" / "run_mfrinstseg_final_v1_seed42.ps1"


def test_final_v1_runner_tests_each_experiments_best_miou_checkpoint(tmp_path):
    raw = tmp_path / "raw"
    parent = tmp_path / "parent"
    shared = tmp_path / "shared"
    (shared / "bin").mkdir(parents=True)
    raw.mkdir()
    parent.mkdir()
    prepared = tmp_path / "prepared"
    results = tmp_path / "results"
    events = tmp_path / "events.txt"

    subset_builder = tmp_path / "fake_subset.py"
    subset_builder.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--raw-dataset')
parser.add_argument('--parent-prepared-dataset')
parser.add_argument('--output-prepared-dataset', required=True)
parser.add_argument('--shared-bin-cache')
parser.add_argument('--sample-percent')
parser.add_argument('--seed')
parser.add_argument('--rebuild', action='store_true')
args = parser.parse_args()
root = Path(args.output_prepared_dataset)
root.mkdir(parents=True, exist_ok=True)
for name in ('train.txt', 'val.txt', 'test.txt'):
    (root / name).write_text('sample\\n', encoding='utf-8')
(root / 'preparation_manifest.json').write_text(
    json.dumps({'seed': 42}), encoding='utf-8'
)
""".strip()
        + "\n",
        encoding="utf-8",
    )

    fake_train = tmp_path / "fake_train.ps1"
    fake_train.write_text(
        r'''
param(
    [string]$Mode,
    [string]$InstanceRelationMapping,
    [string]$PreparedDatasetPath,
    [string]$ResultBasePath,
    [string]$SurfaceNeighborMetric,
    [bool]$SurfaceTrimAware,
    [string]$SurfaceFusion,
    [int]$Seed,
    [Parameter(ValueFromRemainingArguments=$true)]$Remaining
)
$trim = if ($SurfaceTrimAware) { "trim" } else { "untrimmed" }
$mapping = if ($InstanceRelationMapping -eq "positive_only") { "positive" } else { $InstanceRelationMapping }
switch ($Mode) {
    "topo_aag" { $label = "topo_aag" }
    "uv_aag" { $label = "uv_aag" }
    "pc_aag" { $label = "pc_aag_param_surface_${SurfaceNeighborMetric}_${trim}_${SurfaceFusion}" }
    "cyl_aag" { $label = "cyl_aag" }
    "spe_pcaag" { $label = "spe_pcaag_param_surface_${SurfaceNeighborMetric}_${trim}_${SurfaceFusion}" }
    default { $label = "${Mode}_${mapping}_param_surface_${SurfaceNeighborMetric}_${trim}_${SurfaceFusion}" }
}
$run = Join-Path (Join-Path $ResultBasePath "MFRInstSeg_1pct_${label}_seed${Seed}") "fake-run"
New-Item -ItemType Directory -Path $run -Force | Out-Null
$miou = Join-Path $run "best-mIoU-010-0.800000.ckpt"
$macro = Join-Path $run "best-Macro-F1-011-0.900000.ckpt"
[System.IO.File]::WriteAllText($miou, "miou")
[System.IO.File]::WriteAllText($macro, "macro")
[System.IO.File]::WriteAllText((Join-Path $run "last.ckpt"), "last")
(Get-Item -LiteralPath $miou).LastWriteTimeUtc = [datetime]"2020-01-01T00:00:00Z"
(Get-Item -LiteralPath $macro).LastWriteTimeUtc = [datetime]"2021-01-01T00:00:00Z"
'''.strip()
        + "\n",
        encoding="utf-8",
    )

    fake_test = tmp_path / "fake_test.ps1"
    fake_test.write_text(
        r'''
param(
    [string]$Checkpoint,
    [Parameter(ValueFromRemainingArguments=$true)]$Remaining
)
[System.IO.File]::AppendAllText(
    $env:FINAL_V1_CHECKPOINT_EVENTS,
    "$Checkpoint`n"
)
'''.strip()
        + "\n",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["FINAL_V1_CHECKPOINT_EVENTS"] = str(events)
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(RUNNER),
            "-RawDatasetPath",
            str(raw),
            "-ParentPreparedDatasetPath",
            str(parent),
            "-PreparedDatasetPath",
            str(prepared),
            "-SharedBinCachePath",
            str(shared),
            "-ResultBasePath",
            str(results),
            "-PythonExe",
            os.sys.executable,
            "-SubsetBuilder",
            str(subset_builder),
            "-TrainingLauncher",
            str(fake_train),
            "-TestLauncher",
            str(fake_test),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    selected = events.read_text(encoding="utf-8").splitlines()
    assert len(selected) == 7
    assert all(Path(path).name.startswith("best-mIoU-") for path in selected)
