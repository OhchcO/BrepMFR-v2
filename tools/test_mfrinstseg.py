#!/usr/bin/env python
"""Evaluate a trained BrepMFR checkpoint on an existing BIN split."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import CADSynth
from models.brepseg_model import BrepSeg


@dataclass(frozen=True)
class GeometryRequirements:
    """Strict BIN fields selected by checkpoint-saved model switches."""

    require_face_pointcloud: bool
    require_cylindrical_pos: bool


def _configuration_value(configuration, name, default):
    if isinstance(configuration, Mapping):
        return configuration.get(name, default)
    return getattr(configuration, name, default)


def geometry_requirements(hparams):
    """Translate checkpoint hyperparameters into strict dataset requirements."""
    return GeometryRequirements(
        require_face_pointcloud=bool(
            _configuration_value(hparams, "use_face_pointcloud", False)
        ),
        require_cylindrical_pos=bool(
            _configuration_value(hparams, "use_cylindrical_pos", False)
        ),
    )


def requires_instance_for_test(experiment_mode):
    """Only oracle inference consumes ground-truth instance annotations."""
    return experiment_mode == "oracle"


def seed_everything(seed):
    """Use the same project-wide deterministic seed setup as training."""
    from utils.reproducibility import seed_everything as seed_project

    seed_project(seed)


def _model_configuration_value(model, name, default):
    if hasattr(model, name):
        return getattr(model, name)
    return _configuration_value(getattr(model, "hparams", {}), name, default)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path(r"D:\Code\Datasets\MFRInstSeg_BrepMFR"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help="Use val only for a smoke test; formal evaluation uses test",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=25)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: checkpoint directory",
    )
    return parser.parse_args()


def select_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch, device):
    moved = {}
    for name, value in batch.items():
        if hasattr(value, "to"):
            moved[name] = value.to(device)
        else:
            moved[name] = value
    return moved


def optional_float(value):
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def metrics_from_confusion(confusion):
    confusion = confusion.astype(np.float64, copy=False)
    true_positive = np.diag(confusion)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    union = support + predicted - true_positive
    f1_denominator = support + predicted

    class_accuracy = np.divide(
        true_positive,
        support,
        out=np.full_like(true_positive, np.nan),
        where=support > 0,
    )
    class_iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan),
        where=union > 0,
    )
    class_f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.full_like(true_positive, np.nan),
        where=f1_denominator > 0,
    )
    return support, class_accuracy, class_iou, class_f1


def write_results(output_dir, result):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "test_metrics.json"
    log_path = output_dir / "test_log.txt"

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "BrepMFR test result",
        "checkpoint={}".format(result["checkpoint"]),
        "checkpoint_epoch={}".format(result["checkpoint_epoch"]),
        "experiment_mode={}".format(result["experiment_mode"]),
        "seed={}".format(result["seed"]),
        "prepared_dataset_path={}".format(result["prepared_dataset_path"]),
        "use_face_pointcloud={}".format(result["use_face_pointcloud"]),
        "use_cylindrical_pos={}".format(result["use_cylindrical_pos"]),
        "pointcloud_scale={}".format(result["pointcloud_scale"]),
        "cylindrical_scale={}".format(result["cylindrical_scale"]),
        "require_instance={}".format(result["require_instance"]),
        "batch_size={}".format(result["batch_size"]),
        "num_workers={}".format(result["num_workers"]),
        "device={}".format(result["device"]),
        "split={}".format(result["split"]),
        "samples={}".format(result["samples"]),
        "faces={}".format(result["faces"]),
        "elapsed_seconds={:.3f}".format(result["elapsed_seconds"]),
        "test_loss={:.8f}".format(result["test_loss"]),
        "per_face_accuracy={:.8f}".format(result["per_face_accuracy"]),
        "mean_iou={:.8f}".format(result["mean_iou"]),
        "macro_f1={:.8f}".format(result["macro_f1"]),
        "",
        "per_class_metrics:",
    ]
    for item in result["per_class"]:
        lines.append(
            "class_{class_id:02d} support={support} accuracy={accuracy} "
            "iou={iou} f1={f1}".format(
                class_id=item["class_id"],
                support=item["support"],
                accuracy=(
                    "{:.8f}".format(item["accuracy"])
                    if item["accuracy"] is not None
                    else "NA"
                ),
                iou=(
                    "{:.8f}".format(item["iou"])
                    if item["iou"] is not None
                    else "NA"
                ),
                f1=(
                    "{:.8f}".format(item["f1"])
                    if item["f1"] is not None
                    else "NA"
                ),
            )
        )
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path, json_path


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(args.checkpoint))
    manifest = args.dataset_path / (args.split + ".txt")
    if not manifest.is_file():
        raise FileNotFoundError(
            "Prepared BIN manifest not found: {}. Prepare the shared dataset cache first.".format(
                manifest
            )
        )

    seed_everything(args.seed)
    device = select_device(args.device)

    checkpoint_document = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_epoch = (
        int(checkpoint_document.get("epoch", -1)) + 1
        if isinstance(checkpoint_document, Mapping)
        else -1
    )
    model = BrepSeg.load_from_checkpoint(
        str(args.checkpoint),
        map_location="cpu",
    )
    mode = _model_configuration_value(model, "experiment_mode", "baseline")
    requirements = geometry_requirements(model)
    pointcloud_scale = float(
        _model_configuration_value(model, "pointcloud_scale", 1.0)
    )
    cylindrical_scale = float(
        _model_configuration_value(model, "cylindrical_scale", 1.0)
    )
    require_instance = requires_instance_for_test(mode)
    dataset = CADSynth(
        root_dir=args.dataset_path,
        split=args.split,
        random_rotate=False,
        num_class=args.num_classes,
        require_instance=require_instance,
        require_face_pointcloud=requirements.require_face_pointcloud,
        require_cylindrical_pos=requirements.require_cylindrical_pos,
        seed=args.seed,
    )
    if len(dataset) == 0:
        raise RuntimeError("No BIN files were loaded for split {}".format(args.split))
    loader = dataset.get_dataloader(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed_purpose="test_loader",
    )

    model.to(device)
    model.eval()
    confusion = torch.zeros(
        (args.num_classes, args.num_classes), dtype=torch.long
    )
    loss_sum = 0.0
    face_count = 0
    started_at = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            probabilities = model(batch)
            labels = batch["label_feature"].long()
            predictions = probabilities.argmax(dim=-1)
            loss_sum += float(
                (
                    -torch.log(
                        probabilities.gather(1, labels.unsqueeze(1)).clamp_min(1e-12)
                    )
                ).sum().item()
            )
            face_count += int(labels.numel())
            indices = (
                labels * args.num_classes + predictions
            ).detach().cpu()
            confusion += torch.bincount(
                indices,
                minlength=args.num_classes * args.num_classes,
            ).reshape(args.num_classes, args.num_classes)

    elapsed_seconds = time.perf_counter() - started_at
    confusion_np = confusion.numpy()
    support, class_accuracy, class_iou, class_f1 = metrics_from_confusion(
        confusion_np
    )
    correct = float(np.diag(confusion_np).sum())
    per_class = []
    for class_id in range(args.num_classes):
        per_class.append(
            {
                "class_id": class_id,
                "support": int(support[class_id]),
                "accuracy": optional_float(class_accuracy[class_id]),
                "iou": optional_float(class_iou[class_id]),
                "f1": optional_float(class_f1[class_id]),
            }
        )

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "experiment_mode": mode,
        "seed": args.seed,
        "prepared_dataset_path": str(args.dataset_path.resolve()),
        "use_face_pointcloud": requirements.require_face_pointcloud,
        "use_cylindrical_pos": requirements.require_cylindrical_pos,
        "pointcloud_scale": pointcloud_scale,
        "cylindrical_scale": cylindrical_scale,
        "require_instance": require_instance,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "split": args.split,
        "samples": len(dataset),
        "faces": face_count,
        "elapsed_seconds": elapsed_seconds,
        "test_loss": loss_sum / max(face_count, 1),
        "per_face_accuracy": correct / max(face_count, 1),
        "mean_iou": float(np.nanmean(class_iou)),
        "macro_f1": float(np.nanmean(class_f1)),
        "per_class": per_class,
        "confusion_matrix": confusion_np.astype(int).tolist(),
    }
    output_dir = args.output_dir or args.checkpoint.parent
    log_path, json_path = write_results(output_dir, result)
    print("Test completed: {}".format(args.split))
    print("Samples: {} Faces: {}".format(result["samples"], result["faces"]))
    print("Accuracy: {:.4%}".format(result["per_face_accuracy"]))
    print("mIoU: {:.4%}".format(result["mean_iou"]))
    print("Macro-F1: {:.4%}".format(result["macro_f1"]))
    print("Log: {}".format(log_path))
    print("JSON: {}".format(json_path))


if __name__ == "__main__":
    main()
