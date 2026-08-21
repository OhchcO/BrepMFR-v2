"""End-to-end STEP inference orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Dict, Optional

import numpy as np
import torch

from brepmfr.inference.cache import (
    build_cache_identity,
    inspect_cache,
    sha256_file,
    write_cache_metadata_atomic,
)
from brepmfr.inference.conversion import (
    conversion_options_from_requirements,
    convert_step_to_inference_bin,
    default_conversion_implementation_paths,
    validate_inference_bin,
)
from brepmfr.inference.loading import load_inference_batch, select_device
from brepmfr.inference.postprocess import (
    GeometryRequirements,
    configuration_value,
    geometry_requirements,
    postprocess_predictions,
)
from brepmfr.inference.result import (
    build_error_document,
    build_success_document,
    write_json_atomic,
)
from models.brepseg_model import BrepSeg
from utils.reproducibility import DEFAULT_SEED


@dataclass(frozen=True)
class InferenceRequest:
    step_path: Path
    checkpoint_path: Path
    output_dir: Path
    device: str = "auto"
    affinity_threshold: float = 0.5
    force_rebuild: bool = False
    seed: Optional[int] = None


@dataclass(frozen=True)
class InferenceOutcome:
    document: Dict[str, object]
    result_path: Path
    cache_hit: bool
    semantic_probabilities: np.ndarray
    affinity_probabilities: np.ndarray


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: BrepSeg
    epoch: int
    experiment_mode: str
    requirements: GeometryRequirements
    seed: int
    num_classes: int


class InferencePipelineError(RuntimeError):
    """A pipeline failure annotated with its public stage and result path."""

    def __init__(self, stage: str, result_path: Optional[Path]):
        self.stage = stage
        self.result_path = result_path
        message = "STEP inference failed during {0}".format(stage)
        if result_path is not None:
            message += "; result: {0}".format(result_path)
        super().__init__(message)


def _load_checkpoint(path: Path) -> LoadedCheckpoint:
    """Load model weights and all inference-relevant stored metadata."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Checkpoint not found: {0}".format(path))
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, Mapping):
        raise ValueError("Checkpoint root must be a mapping")
    configuration = raw.get("hyper_parameters", raw.get("hparams", {}))
    model = BrepSeg.load_from_checkpoint(str(path), map_location="cpu")
    return LoadedCheckpoint(
        model=model,
        epoch=int(raw.get("epoch", -1)),
        experiment_mode=str(
            configuration_value(configuration, "experiment_mode", "baseline")
        ),
        requirements=geometry_requirements(configuration),
        seed=int(configuration_value(configuration, "seed", DEFAULT_SEED)),
        num_classes=int(configuration_value(configuration, "num_classes", 25)),
    )


def _validate_request(request: InferenceRequest) -> tuple:
    step_path = Path(request.step_path).resolve()
    checkpoint_path = Path(request.checkpoint_path).resolve()
    output_dir = Path(request.output_dir).resolve()
    if step_path.suffix.lower() not in (".step", ".stp"):
        raise ValueError("Input must be a .step or .stp file: {0}".format(step_path))
    if not step_path.is_file():
        raise FileNotFoundError("STEP file not found: {0}".format(step_path))
    if checkpoint_path.suffix.lower() != ".ckpt":
        raise ValueError("Checkpoint must use the .ckpt suffix: {0}".format(checkpoint_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {0}".format(checkpoint_path))
    threshold = float(request.affinity_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Affinity threshold must be finite and in [0, 1]")
    if request.seed is not None and isinstance(request.seed, bool):
        raise ValueError("Seed must be an integer")
    return step_path, checkpoint_path, output_dir, threshold


def _numpy_tensor(value: torch.Tensor, name: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise TypeError("{0} must be a torch.Tensor".format(name))
    return value.detach().cpu().numpy()


def run_step_inference(request: InferenceRequest) -> InferenceOutcome:
    """Run checkpoint-driven STEP conversion and feature recognition."""
    stage = "arguments"
    result_path = None
    started = time.perf_counter()
    try:
        output_dir = Path(request.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "result.json"
        step_path, checkpoint_path, output_dir, threshold = _validate_request(
            request
        )

        stage = "checkpoint"
        checkpoint_started = time.perf_counter()
        loaded = _load_checkpoint(checkpoint_path)
        if loaded.experiment_mode not in ("dynamic", "dynamic_affinity"):
            raise ValueError(
                "Checkpoint experiment mode must provide dynamic affinity logits"
            )
        device = select_device(request.device)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        checkpoint_duration = time.perf_counter() - checkpoint_started
        seed = loaded.seed if request.seed is None else int(request.seed)
        conversion_options = conversion_options_from_requirements(
            loaded.requirements, seed
        )

        stage = "cache"
        bin_path = output_dir / "input.bin"
        metadata_path = output_dir / "input.bin.meta.json"
        identity = build_cache_identity(
            step_path,
            default_conversion_implementation_paths(),
            seed,
            asdict(conversion_options.point_sampling),
            {
                "use_face_uv": loaded.requirements.use_face_uv,
                "face_pointcloud": loaded.requirements.require_face_pointcloud,
                "parametric_surface": loaded.requirements.require_parametric_surface,
                "cylindrical_pos": loaded.requirements.require_cylindrical_pos,
            },
        )

        def validate_bin_file(path):
            validate_inference_bin(
                path,
                loaded.requirements,
                conversion_options.point_sampling,
            )

        decision = inspect_cache(
            bin_path=bin_path,
            metadata_path=metadata_path,
            identity=identity,
            validate_bin_file=validate_bin_file,
            force_rebuild=bool(request.force_rebuild),
        )
        conversion_duration = 0.0
        if not decision.hit:
            stage = "conversion"
            conversion_started = time.perf_counter()
            convert_step_to_inference_bin(
                step_path,
                decision.bin_path,
                loaded.requirements,
                seed,
            )
            validate_bin_file(decision.bin_path)
            stage = "cache"
            write_cache_metadata_atomic(
                decision.metadata_path,
                identity,
                decision.fingerprint,
            )
            conversion_duration = time.perf_counter() - conversion_started

        stage = "loading"
        batch = load_inference_batch(
            decision.bin_path,
            loaded.requirements,
            loaded.num_classes,
            seed,
            device,
        )
        padding_mask = batch.get("padding_mask")
        if not isinstance(padding_mask, torch.Tensor) or padding_mask.shape[0] != 1:
            raise ValueError("Inference batch must contain exactly one graph")
        face_count = int((~padding_mask[0].to(torch.bool)).sum().item())

        stage = "inference"
        inference_started = time.perf_counter()
        model = loaded.model.to(device)
        model.eval()
        with torch.inference_mode():
            semantic_tensor, affinity_tensor = model.predict_with_affinity(batch)
        if affinity_tensor is None:
            raise ValueError("Checkpoint did not produce affinity logits")
        semantic_probabilities = _numpy_tensor(
            semantic_tensor, "semantic probabilities"
        )
        if semantic_probabilities.ndim != 2:
            raise ValueError("Semantic probabilities must have shape (N, C)")
        semantic_probabilities = semantic_probabilities[:face_count]
        affinity_logits = _numpy_tensor(affinity_tensor, "affinity logits")
        if affinity_logits.ndim == 3 and affinity_logits.shape[0] == 1:
            affinity_logits = affinity_logits[0]
        if affinity_logits.ndim != 2:
            raise ValueError("Affinity logits must have shape (1, N, N) or (N, N)")
        affinity_logits = affinity_logits[:face_count, :face_count]
        spatial_pos = _numpy_tensor(batch["spatial_pos"], "spatial_pos")
        if spatial_pos.ndim == 3 and spatial_pos.shape[0] == 1:
            spatial_pos = spatial_pos[0]
        spatial_pos = spatial_pos[:face_count, :face_count]
        inference_duration = time.perf_counter() - inference_started

        stage = "postprocess"
        processed = postprocess_predictions(
            semantic_probabilities,
            affinity_logits,
            spatial_pos,
            threshold=threshold,
            background_class_id=24,
        )
        timings = {
            "checkpoint_load": checkpoint_duration,
            "conversion": conversion_duration,
            "inference": inference_duration,
            "total": time.perf_counter() - started,
        }
        document = build_success_document(
            step_path=step_path,
            step_sha256=str(identity["step_sha256"]),
            face_count=face_count,
            bin_path=decision.bin_path,
            metadata_path=decision.metadata_path,
            cache_hit=decision.hit,
            cache_fingerprint=decision.fingerprint,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_epoch=loaded.epoch,
            experiment_mode=loaded.experiment_mode,
            device=str(device),
            num_classes=loaded.num_classes,
            affinity_threshold=threshold,
            faces=processed.faces,
            instances=processed.instances,
            timings_seconds=timings,
            warnings=[],
        )

        stage = "serialization"
        write_json_atomic(result_path, document)
        return InferenceOutcome(
            document=document,
            result_path=result_path,
            cache_hit=decision.hit,
            semantic_probabilities=semantic_probabilities,
            affinity_probabilities=processed.affinity_probabilities,
        )
    except Exception as error:
        failure_stage = stage
        if result_path is not None:
            try:
                error_document = build_error_document(
                    failure_stage,
                    error,
                    Path(request.step_path),
                )
                write_json_atomic(result_path, error_document)
            except Exception as serialization_error:
                raise InferencePipelineError(
                    "serialization", result_path
                ) from serialization_error
        raise InferencePipelineError(failure_stage, result_path) from error

