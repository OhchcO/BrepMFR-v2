"""Load one inference BIN into the existing model batch contract."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from brepmfr.inference.postprocess import GeometryRequirements
from data.collator import collator
from data.dataset import CADSynth


class SingleBinDataset(CADSynth):
    """Manifest-free adapter around ``CADSynth.load_one_graph``."""

    def __init__(
        self,
        bin_path: Path,
        requirements: GeometryRequirements,
        num_classes: int,
        seed: int,
    ) -> None:
        bin_path = Path(bin_path).resolve()
        if not bin_path.is_file():
            raise FileNotFoundError("Inference BIN not found: {0}".format(bin_path))
        self.split = "inference"
        self.num_class = int(num_classes)
        self.random_rotate = False
        self.require_instance = False
        self.require_face_pointcloud = requirements.require_face_pointcloud
        self.require_parametric_surface = requirements.require_parametric_surface
        self.require_cylindrical_pos = requirements.require_cylindrical_pos
        self.allow_unlabeled = True
        self.seed = int(seed)
        self.file_paths = [bin_path]


def select_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic inference device."""
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return torch.device("cuda")
    raise ValueError("Unsupported device: {0}".format(requested))


def load_inference_batch(
    bin_path: Path,
    requirements: GeometryRequirements,
    num_classes: int,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    """Load, collate, validate, and move a single inference graph."""
    dataset = SingleBinDataset(bin_path, requirements, num_classes, seed)
    sample = dataset[0]
    batch = collator(
        [sample],
        multi_hop_max_dist=16,
        spatial_pos_max=32,
    )
    if not bool((batch["label_feature"] == -1).all()):
        raise ValueError("Inference batch must use only semantic sentinel -1")
    if bool(batch["instance_available"].any()):
        raise ValueError("Inference batch must not contain instance ground truth")

    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved
