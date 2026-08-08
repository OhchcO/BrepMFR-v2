"""Read-only environment smoke test for BrepMFR.

The dependency-light helpers in this module intentionally avoid importing the
training stack at module import time.  The real PyTorch/DGL integration is
loaded only when the smoke test runs so missing packages can be reported
clearly.
"""

import argparse
import platform
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


CHECK_LEVELS = ("PASS", "WARN", "FAIL")
SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class CheckResult:
    level: str
    stage: str
    message: str
    hint: str = ""

    def __post_init__(self) -> None:
        if self.level not in CHECK_LEVELS:
            raise ValueError("level must be PASS, WARN, or FAIL")
        if not self.stage:
            raise ValueError("stage must not be empty")
        if not self.message:
            raise ValueError("message must not be empty")

    def render(self) -> str:
        line = "[{0}] {1}: {2}".format(
            self.level,
            self.stage,
            self.message,
        )
        if self.hint:
            line += " | Hint: " + self.hint
        return line


def select_device(requested: str, cuda_available: bool) -> str:
    """Resolve an auto/cpu/cuda request without importing PyTorch."""
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def _manifest_ids(path: Path) -> Tuple[str, ...]:
    ids = []
    seen = set()
    for line_number, raw_id in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        sample_id = raw_id.strip()
        if not sample_id:
            raise ValueError(
                "blank sample ID in {0} at line {1}".format(
                    path,
                    line_number,
                )
            )
        if sample_id in seen:
            raise ValueError(
                "duplicate sample ID {0} in {1}".format(sample_id, path)
            )
        if sample_id in (".", "..") or "/" in sample_id or "\\" in sample_id:
            raise ValueError("unsafe sample ID {0} in {1}".format(sample_id, path))
        seen.add(sample_id)
        ids.append(sample_id)
    return tuple(ids)


def _find_bin(dataset_path: Path, sample_id: str) -> Path:
    matches = sorted(
        path for path in dataset_path.rglob("*.bin") if path.stem == sample_id
    )
    if not matches:
        raise FileNotFoundError(
            "BIN file for sample {0} was not found under {1}".format(
                sample_id,
                dataset_path,
            )
        )
    if len(matches) > 1:
        raise ValueError(
            "Multiple BIN files share sample stem {0}: {1}".format(
                sample_id,
                ", ".join(str(path) for path in matches),
            )
        )
    return matches[0]


def resolve_sample(
    dataset_path: Path,
    sample_id: Optional[str],
) -> Tuple[str, str, Path]:
    """Resolve one manifest member and its unique BIN file."""
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            "Dataset directory does not exist: {0}".format(dataset_path)
        )

    manifests = []
    for split in SPLIT_ORDER:
        manifest = dataset_path / (split + ".txt")
        if manifest.is_file():
            manifests.append((split, manifest, _manifest_ids(manifest)))
    if not manifests:
        raise FileNotFoundError(
            "No split manifest (train.txt, val.txt, or test.txt) exists in {0}".format(
                dataset_path
            )
        )

    if sample_id is not None:
        for split, _manifest, ids in manifests:
            if sample_id in ids:
                return split, sample_id, _find_bin(dataset_path, sample_id)
        raise ValueError(
            "sample {0} is not listed in an available split manifest".format(
                sample_id
            )
        )

    first_missing_error = None
    for split, _manifest, ids in manifests:
        for candidate in ids:
            try:
                return split, candidate, _find_bin(dataset_path, candidate)
            except FileNotFoundError as error:
                if first_missing_error is None:
                    first_missing_error = error
    if first_missing_error is not None:
        raise first_missing_error
    raise ValueError("available split manifests contain no sample IDs")


def validate_output(output: Any, face_count: int, num_classes: int) -> None:
    """Validate the observable semantic-forward contract."""
    expected = (int(face_count), int(num_classes))
    actual = tuple(output.shape)
    if actual != expected:
        raise ValueError(
            "expected output shape {0}, got {1}".format(expected, actual)
        )
    if not bool(output.isfinite().all().item()):
        raise ValueError("model output contains non-finite values")


def exit_code(results: Sequence[CheckResult]) -> int:
    return 1 if any(result.level == "FAIL" for result in results) else 0


@dataclass(frozen=True)
class SmokeTestConfig:
    dataset_path: Path
    device: str = "auto"
    sample_id: Optional[str] = None
    experiment: str = "baseline"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))


def build_model_args(experiment: str) -> Namespace:
    """Build a small model while preserving formal experiment inputs."""
    from segmentation import resolve_experiment_alias

    switches = resolve_experiment_alias(experiment)
    parameter_surface = experiment == "pc_aag"
    return Namespace(
        num_classes=25,
        n_layers_encode=2,
        dim_node=128,
        d_model=128,
        n_heads=8,
        dropout=0.0,
        attention_dropout=0.0,
        act_dropout=0.0,
        experiment_mode=switches.experiment_mode,
        use_face_uv=switches.use_face_uv,
        use_face_pointcloud=switches.use_face_pointcloud,
        use_cylindrical_pos=switches.use_cylindrical_pos,
        instance_bias_scale=switches.instance_bias_scale,
        instance_bias_strategy=switches.instance_bias_strategy,
        instance_relation_mapping="positive_only",
        instance_bias_last_layers=2,
        instance_bias_confidence_power=1.0,
        instance_bias_gate_init=0.0,
        instance_bias_max_gate_scale=1.0,
        instance_bias_delay_epochs=10,
        instance_bias_ramp_epochs=10,
        affinity_dim=64,
        affinity_temperature=0.2,
        affinity_loss_weight=0.1,
        affinity_warmup_epochs=10,
        pointcloud_scale=1.0,
        cylindrical_scale=1.0,
        face_geometry_encoder=(
            "param_surface" if parameter_surface else "mini_point"
        ),
        surface_neighbor_metric="intrinsic",
        surface_trim_aware=False,
        surface_fusion="gated" if parameter_surface else "add",
    )


def move_batch(batch: Mapping[str, Any], device: str) -> Dict[str, Any]:
    """Move tensor/DGL values while preserving identifiers and metadata."""
    moved = {}
    for name, value in batch.items():
        if value is not None and hasattr(value, "to"):
            moved[name] = value.to(device)
        else:
            moved[name] = value
    return moved


def _load_components() -> Dict[str, Any]:
    import dgl
    import numpy
    import pytorch_lightning
    import torch

    from data.collator import collator
    from data.dataset import CADSynth
    from models.brepseg_model import BrepSeg

    return {
        "torch": torch,
        "versions": {
            "DGL": dgl.__version__,
            "NumPy": numpy.__version__,
            "Lightning": pytorch_lightning.__version__,
        },
        "dataset_class": CADSynth,
        "collator": collator,
        "model_class": BrepSeg,
    }


def _failure(stage: str, error: Exception, hint: str) -> CheckResult:
    return CheckResult(
        "FAIL",
        stage,
        "{0}: {1}".format(type(error).__name__, error),
        hint,
    )


def run_forward_smoke_test(
    config: SmokeTestConfig,
    components: Optional[Mapping[str, Any]] = None,
) -> list:
    """Execute one real read-only data-to-model forward smoke test."""
    results = [
        CheckResult(
            "PASS",
            "Python",
            "{0} ({1})".format(platform.python_version(), sys.executable),
        )
    ]

    try:
        loaded = dict(components) if components is not None else _load_components()
        torch = loaded["torch"]
        versions = loaded["versions"]
        version_text = ", ".join(
            ["PyTorch {0}".format(torch.__version__)]
            + ["{0} {1}".format(name, value) for name, value in versions.items()]
        )
        results.append(CheckResult("PASS", "Dependencies", version_text))
    except Exception as error:
        results.append(
            _failure(
                "Dependencies",
                error,
                "Activate the aagmspenv Conda environment and install environment.yml.",
            )
        )
        return results

    try:
        device = select_device(config.device, bool(torch.cuda.is_available()))
        if device == "cuda":
            name = torch.cuda.get_device_name(0)
            properties = torch.cuda.get_device_properties(0)
            memory_gib = float(properties.total_memory) / (1024.0 ** 3)
            message = "cuda:0, {0}, {1:.2f} GiB".format(name, memory_gib)
            results.append(CheckResult("PASS", "Device", message))
        else:
            level = "WARN" if config.device == "auto" else "PASS"
            message = "CPU selected"
            hint = "Use --device cuda to require GPU execution." if level == "WARN" else ""
            results.append(CheckResult(level, "Device", message, hint))
    except Exception as error:
        results.append(
            _failure(
                "Device",
                error,
                "Use --device cpu or install a CUDA-enabled PyTorch build.",
            )
        )
        return results

    try:
        split, sample_id, bin_path = resolve_sample(
            config.dataset_path,
            config.sample_id,
        )
        results.append(
            CheckResult(
                "PASS",
                "Dataset",
                "{0} sample {1} ({2})".format(split, sample_id, bin_path),
            )
        )
    except Exception as error:
        results.append(
            _failure(
                "Dataset",
                error,
                "Pass a prepared BIN cache containing split manifests.",
            )
        )
        return results

    try:
        args = build_model_args(config.experiment)
        dataset = loaded["dataset_class"](
            root_dir=config.dataset_path,
            split=split,
            random_rotate=False,
            num_class=args.num_classes,
            require_instance=args.experiment_mode == "oracle",
            require_face_pointcloud=args.use_face_pointcloud,
            require_parametric_surface=(
                args.use_face_pointcloud
                and args.face_geometry_encoder == "param_surface"
            ),
            require_cylindrical_pos=args.use_cylindrical_pos,
            seed=42,
        )
        matching_indices = [
            index
            for index, path in enumerate(dataset.file_paths)
            if Path(path).resolve() == bin_path.resolve()
        ]
        if len(matching_indices) != 1:
            raise ValueError(
                "resolved sample was not uniquely present in CADSynth"
            )
        item = dataset[matching_indices[0]]
        batch = loaded["collator"](
            [item],
            multi_hop_max_dist=16,
            spatial_pos_max=32,
        )
        results.append(
            CheckResult(
                "PASS",
                "Data pipeline",
                "CADSynth and collator loaded sample {0}".format(sample_id),
            )
        )
    except Exception as error:
        results.append(
            _failure(
                "Data pipeline",
                error,
                "Check BIN schema and the fields required by --experiment.",
            )
        )
        return results

    try:
        model = loaded["model_class"](args).eval().to(device)
        batch = move_batch(batch, device)
        with torch.no_grad():
            output = model(batch)
        face_count = int((~batch["padding_mask"]).sum().item())
        validate_output(output, face_count, args.num_classes)
        results.append(
            CheckResult(
                "PASS",
                "Forward",
                "output shape {0} on {1}".format(tuple(output.shape), device),
            )
        )
    except Exception as error:
        results.append(
            _failure(
                "Forward",
                error,
                "Check model dependencies, device compatibility, and BIN feature shapes.",
            )
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the BrepMFR environment with one real prepared BIN "
            "and a no-gradient model forward pass."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Prepared MFRInstSeg BIN dataset containing split manifests.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device policy (default: auto).",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Optional exact sample ID from train/val/test manifests.",
    )
    parser.add_argument(
        "--experiment",
        default="baseline",
        help=(
            "Existing experiment alias (default: baseline). Use pc_aag to "
            "check the formal parameter-surface path."
        ),
    )
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    runner: Callable[[SmokeTestConfig], Sequence[CheckResult]] = (
        run_forward_smoke_test
    ),
) -> int:
    args = build_parser().parse_args(argv)
    config = SmokeTestConfig(
        dataset_path=args.dataset_path,
        device=args.device,
        sample_id=args.sample_id,
        experiment=args.experiment,
    )

    print("BrepMFR environment smoke test")
    print("Dataset: {0}".format(config.dataset_path.resolve()))
    print("Experiment/device: {0} / {1}".format(
        config.experiment,
        config.device,
    ))
    print("")
    try:
        results = list(runner(config))
    except Exception as error:
        results = [
            _failure(
                "Smoke test",
                error,
                "Review the exception and rerun in the aagmspenv environment.",
            )
        ]
    for result in results:
        print(result.render())

    code = exit_code(results)
    print("")
    print("ENVIRONMENT READY" if code == 0 else "ENVIRONMENT NOT READY")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
