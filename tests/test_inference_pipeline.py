import json
from pathlib import Path

import numpy as np
import pytest
import torch

from brepmfr.inference.cache import CacheDecision
from brepmfr.inference.conversion import ConversionArtifacts
from brepmfr.inference.face_signature import face_mapping_json_bytes
from brepmfr.inference.pipeline import (
    InferenceOutcome,
    InferencePipelineError,
    InferenceRequest,
    LoadedCheckpoint,
)
from brepmfr.inference.postprocess import GeometryRequirements
from brepmfr.inference import pipeline
from tools import infer_step as cli


class FakeModel:
    def __init__(self, calls, affinity=True):
        self.calls = calls
        self.affinity = affinity

    def to(self, device):
        self.calls.append(("model-to", str(device)))
        return self

    def eval(self):
        self.calls.append("model-eval")
        return self

    def predict_with_affinity(self, batch):
        self.calls.append("predict")
        probabilities = torch.zeros((2, 25), dtype=torch.float32)
        probabilities[:, 1] = 1.0
        affinity_logits = (
            torch.full((1, 2, 2), 10.0, dtype=torch.float32)
            if self.affinity
            else None
        )
        return probabilities, affinity_logits


def _two_face_mapping():
    faces = []
    for face_index in range(2):
        faces.append(
            {
                "face_index": face_index,
                "surface_type": "plane",
                "area": 1.0,
                "perimeter": 4.0,
                "centroid": [float(face_index), 0.0, 0.0],
                "bounding_box": [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                "loop_count": 1,
                "edge_count": 4,
                "neighbor_count": 1,
                "surface_parameters": {
                    "origin": [0.0, 0.0, 0.0],
                    "axis": [0.0, 0.0, 1.0],
                },
                "sample_points": [[float(face_index), 0.0, 0.0]],
            }
        )
    return {
        "contract_version": "1.0",
        "status": "ready",
        "normalization": {
            "kind": "aag_unit_box",
            "box_side": 1.0,
            "scale_rule": "2_over_longest_bbox_extent",
        },
        "faces": faces,
        "adjacency_edges": [[0, 1]],
    }


def make_request_and_patch_runtime(tmp_path, monkeypatch):
    step = tmp_path / "part.step"
    checkpoint = tmp_path / "model.ckpt"
    step.write_bytes(b"STEP")
    checkpoint.write_bytes(b"CHECKPOINT")
    request = InferenceRequest(
        step,
        checkpoint,
        tmp_path / "output",
        device="cpu",
    )
    calls = []
    captured = {}
    loaded = LoadedCheckpoint(
        model=FakeModel(calls),
        epoch=187,
        experiment_mode="dynamic_affinity",
        requirements=GeometryRequirements(False, False, False, False),
        seed=42,
        num_classes=25,
    )
    monkeypatch.setattr(
        pipeline,
        "_load_checkpoint",
        lambda path: calls.append("checkpoint") or loaded,
    )
    monkeypatch.setattr(
        pipeline,
        "select_device",
        lambda requested: calls.append(("device", requested))
        or torch.device("cpu"),
    )
    monkeypatch.setattr(
        pipeline,
        "default_conversion_implementation_paths",
        lambda: (checkpoint,),
    )

    def fake_identity(
        step_path,
        implementation_paths,
        seed,
        point_sampling,
        geometry_fields,
    ):
        calls.append("identity")
        captured["seed"] = seed
        captured["point_sampling"] = point_sampling
        captured["geometry_fields"] = geometry_fields
        return {"step_sha256": "step-hash", "seed": seed}

    monkeypatch.setattr(pipeline, "build_cache_identity", fake_identity)
    monkeypatch.setattr(
        pipeline,
        "load_inference_batch",
        lambda *args, **kwargs: calls.append(("load", args[3]))
        or {
            "padding_mask": torch.tensor([[False, False]]),
            "spatial_pos": torch.tensor([[[1, 2], [2, 1]]]),
        },
    )
    monkeypatch.setattr(
        pipeline,
        "sha256_file",
        lambda path: "checkpoint-hash"
        if Path(path).suffix == ".ckpt"
        else "step-hash",
    )
    monkeypatch.setattr(
        pipeline,
        "write_json_atomic",
        lambda path, document: Path(path).write_text(
            json.dumps(document), encoding="utf-8"
        ),
    )
    return request, loaded, calls, captured


def test_load_checkpoint_uses_stored_hyperparameters(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    raw = {
        "epoch": 187,
        "hyper_parameters": {
            "experiment_mode": "dynamic_affinity",
            "num_classes": 25,
            "seed": 42,
            "use_face_uv": False,
            "use_face_pointcloud": True,
            "face_geometry_encoder": "param_surface",
            "use_cylindrical_pos": True,
        },
    }
    model = object()
    load_calls = []
    monkeypatch.setattr(
        pipeline.torch,
        "load",
        lambda path, map_location: load_calls.append((path, map_location)) or raw,
    )
    monkeypatch.setattr(
        pipeline.BrepSeg,
        "load_from_checkpoint",
        lambda path, map_location: model,
    )

    loaded = pipeline._load_checkpoint(checkpoint)

    assert load_calls == [(checkpoint.resolve(), "cpu")]
    assert loaded.model is model
    assert loaded.epoch == 187
    assert loaded.experiment_mode == "dynamic_affinity"
    assert loaded.seed == 42
    assert loaded.num_classes == 25
    assert loaded.requirements == GeometryRequirements(False, True, True, True)


def test_pipeline_reuses_valid_bin_and_skips_conversion(tmp_path, monkeypatch):
    request, _, calls, _ = make_request_and_patch_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "inspect_cache",
        lambda **kwargs: calls.append("inspect")
        or CacheDecision(
            True,
            "fp",
            request.output_dir / "input.bin",
            request.output_dir / "input.face_signatures.json",
            request.output_dir / "input.bin.meta.json",
            "hit",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "convert_step_to_inference_artifacts",
        lambda *args, **kwargs: calls.append("convert"),
    )
    monkeypatch.setattr(
        pipeline,
        "write_cache_metadata_atomic",
        lambda *args, **kwargs: calls.append("metadata"),
    )

    outcome = pipeline.run_step_inference(request)

    assert "convert" not in calls
    assert "metadata" not in calls
    assert outcome.cache_hit is True
    assert outcome.document["cache"]["hit"] is True
    assert outcome.document["input"]["face_count"] == 2
    assert outcome.document["instances"][0]["face_indices"] == [0, 1]
    assert outcome.semantic_probabilities.shape == (2, 25)
    assert outcome.affinity_probabilities.shape == (2, 2)


def test_pipeline_reports_cache_hit_stage_boundaries_in_order(
    tmp_path, monkeypatch
):
    request, _, _, _ = make_request_and_patch_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "inspect_cache",
        lambda **kwargs: CacheDecision(
            True,
            "fp",
            request.output_dir / "input.bin",
            request.output_dir / "input.face_signatures.json",
            request.output_dir / "input.bin.meta.json",
            "hit",
        ),
    )
    stages = []

    pipeline.run_step_inference(request, progress=stages.append)

    assert stages == [
        "checkpoint",
        "cache",
        "loading",
        "inference",
        "postprocess",
        "serialization",
    ]


def test_pipeline_cache_miss_converts_validates_and_records_seed(
    tmp_path, monkeypatch
):
    request, _, calls, captured = make_request_and_patch_runtime(
        tmp_path, monkeypatch
    )
    bin_path = request.output_dir / "input.bin"
    signature_path = request.output_dir / "input.face_signatures.json"
    metadata_path = request.output_dir / "input.bin.meta.json"
    monkeypatch.setattr(
        pipeline,
        "inspect_cache",
        lambda **kwargs: calls.append("inspect")
        or CacheDecision(
            False,
            "fp",
            bin_path,
            signature_path,
            metadata_path,
            "missing-bin",
        ),
    )

    def convert(step, target, signatures, requirements, seed):
        calls.append(("convert", seed))
        target.write_bytes(b"bin")
        mapping = _two_face_mapping()
        signatures.write_bytes(face_mapping_json_bytes(mapping))
        return ConversionArtifacts(2, mapping)

    monkeypatch.setattr(pipeline, "convert_step_to_inference_artifacts", convert)
    monkeypatch.setattr(
        pipeline,
        "validate_inference_bin",
        lambda *args, **kwargs: calls.append("validate"),
    )
    monkeypatch.setattr(
        pipeline,
        "write_cache_metadata_atomic",
        lambda *args: calls.append("metadata"),
    )

    outcome = pipeline.run_step_inference(request)

    assert captured["seed"] == 42
    assert ("convert", 42) in calls
    assert calls.index("inspect") < calls.index(("convert", 42))
    assert calls.index(("convert", 42)) < calls.index("validate")
    assert calls.index("validate") < calls.index("metadata")
    assert outcome.cache_hit is False


def test_pipeline_explicit_seed_overrides_checkpoint_seed(tmp_path, monkeypatch):
    request, _, _, captured = make_request_and_patch_runtime(tmp_path, monkeypatch)
    request = InferenceRequest(
        request.step_path,
        request.checkpoint_path,
        request.output_dir,
        device="cpu",
        seed=7,
    )
    monkeypatch.setattr(
        pipeline,
        "inspect_cache",
        lambda **kwargs: CacheDecision(
            True,
            "fp",
            request.output_dir / "input.bin",
            request.output_dir / "input.face_signatures.json",
            request.output_dir / "input.bin.meta.json",
            "hit",
        ),
    )

    pipeline.run_step_inference(request)

    assert captured["seed"] == 7


def test_pipeline_returns_absolute_result_path_for_new_relative_output(
    tmp_path, monkeypatch
):
    request, _, _, _ = make_request_and_patch_runtime(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    request = InferenceRequest(
        request.step_path,
        request.checkpoint_path,
        Path("new-relative-output"),
        device="cpu",
    )

    def cache_hit(**kwargs):
        return CacheDecision(
            True,
            "fp",
            kwargs["bin_path"],
            kwargs["signature_path"],
            kwargs["metadata_path"],
            "hit",
        )

    monkeypatch.setattr(pipeline, "inspect_cache", cache_hit)

    outcome = pipeline.run_step_inference(request)

    assert outcome.result_path.is_absolute()
    assert outcome.result_path == (
        tmp_path / "new-relative-output" / "result.json"
    ).resolve()


@pytest.mark.parametrize(
    ("boundary", "expected_stage"),
    [
        ("checkpoint", "checkpoint"),
        ("cache", "cache"),
        ("conversion", "conversion"),
        ("loading", "loading"),
        ("postprocess", "postprocess"),
    ],
)
def test_pipeline_writes_stage_specific_error_documents(
    tmp_path, monkeypatch, boundary, expected_stage
):
    request, _, _, _ = make_request_and_patch_runtime(tmp_path, monkeypatch)
    bin_path = request.output_dir / "input.bin"
    signature_path = request.output_dir / "input.face_signatures.json"
    metadata_path = request.output_dir / "input.bin.meta.json"
    if boundary == "checkpoint":
        monkeypatch.setattr(
            pipeline,
            "_load_checkpoint",
            lambda path: (_ for _ in ()).throw(RuntimeError("checkpoint failed")),
        )
    elif boundary == "cache":
        monkeypatch.setattr(
            pipeline,
            "inspect_cache",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("cache failed")),
        )
    else:
        monkeypatch.setattr(
            pipeline,
            "inspect_cache",
            lambda **kwargs: CacheDecision(
                boundary != "conversion",
                "fp",
                bin_path,
                signature_path,
                metadata_path,
                "hit" if boundary != "conversion" else "missing-bin",
            ),
        )
    if boundary == "conversion":
        monkeypatch.setattr(
            pipeline,
            "convert_step_to_inference_artifacts",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("conversion failed")
            ),
        )
    elif boundary == "loading":
        monkeypatch.setattr(
            pipeline,
            "load_inference_batch",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("loading failed")
            ),
        )
    elif boundary == "postprocess":
        monkeypatch.setattr(
            pipeline,
            "postprocess_predictions",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("postprocess failed")
            ),
        )

    with pytest.raises(InferencePipelineError) as caught:
        pipeline.run_step_inference(request)

    assert caught.value.stage == expected_stage
    document = json.loads((request.output_dir / "result.json").read_text())
    assert document["status"] == "error"
    assert document["stage"] == expected_stage


def test_pipeline_rejects_checkpoint_without_dynamic_affinity(
    tmp_path, monkeypatch
):
    request, loaded, _, _ = make_request_and_patch_runtime(tmp_path, monkeypatch)
    unsupported = LoadedCheckpoint(
        model=loaded.model,
        epoch=loaded.epoch,
        experiment_mode="baseline",
        requirements=loaded.requirements,
        seed=loaded.seed,
        num_classes=loaded.num_classes,
    )
    monkeypatch.setattr(pipeline, "_load_checkpoint", lambda path: unsupported)

    with pytest.raises(InferencePipelineError) as caught:
        pipeline.run_step_inference(request)

    assert caught.value.stage == "checkpoint"
    assert "dynamic" in str(caught.value.__cause__)


def test_pipeline_rejects_missing_affinity_logits_as_inference_error(
    tmp_path, monkeypatch
):
    request, loaded, calls, _ = make_request_and_patch_runtime(tmp_path, monkeypatch)
    no_affinity = LoadedCheckpoint(
        model=FakeModel(calls, affinity=False),
        epoch=loaded.epoch,
        experiment_mode=loaded.experiment_mode,
        requirements=loaded.requirements,
        seed=loaded.seed,
        num_classes=loaded.num_classes,
    )
    monkeypatch.setattr(pipeline, "_load_checkpoint", lambda path: no_affinity)
    monkeypatch.setattr(
        pipeline,
        "inspect_cache",
        lambda **kwargs: CacheDecision(
            True,
            "fp",
            request.output_dir / "input.bin",
            request.output_dir / "input.face_signatures.json",
            request.output_dir / "input.bin.meta.json",
            "hit",
        ),
    )

    with pytest.raises(InferencePipelineError) as caught:
        pipeline.run_step_inference(request)

    assert caught.value.stage == "inference"
    assert "affinity" in str(caught.value.__cause__)


def test_cli_success_prints_summary_and_returns_zero(tmp_path, monkeypatch, capsys):
    step = tmp_path / "part.step"
    checkpoint = tmp_path / "model.ckpt"
    output = tmp_path / "output"
    step.write_bytes(b"STEP")
    checkpoint.write_bytes(b"CHECKPOINT")
    document = {
        "faces": [{"face_index": 0}],
        "instances": [],
        "cache": {"hit": True},
    }
    def fake_inference(request, progress=None):
        assert progress is not None
        progress("checkpoint")
        progress("cache")
        progress("inference")
        return InferenceOutcome(
            document,
            output / "result.json",
            True,
            np.zeros((1, 25)),
            np.zeros((1, 1)),
        )

    monkeypatch.setattr(cli, "run_step_inference", fake_inference)

    exit_code = cli.main(
        [
            "--step",
            str(step),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
        ]
    )

    assert exit_code == 0
    output_text = capsys.readouterr().out
    assert "result.json" in output_text
    assert "Faces: 1" in output_text
    assert "cache: hit" in output_text
    progress_lines = [
        line for line in output_text.splitlines()
        if line.startswith("BREPMFR_PROGRESS:")
    ]
    assert progress_lines == [
        "BREPMFR_PROGRESS:startup",
        "BREPMFR_PROGRESS:importing",
        "BREPMFR_PROGRESS:checkpoint",
        "BREPMFR_PROGRESS:cache",
        "BREPMFR_PROGRESS:inference",
        "BREPMFR_PROGRESS:complete",
    ]


def test_cli_reports_pipeline_error_and_returns_one(
    tmp_path, monkeypatch, capsys
):
    step = tmp_path / "part.step"
    checkpoint = tmp_path / "model.ckpt"
    output = tmp_path / "output"
    step.write_bytes(b"STEP")
    checkpoint.write_bytes(b"CHECKPOINT")

    def fail(_request, progress=None):
        assert progress is not None
        try:
            raise ValueError("model failed")
        except ValueError as error:
            raise InferencePipelineError("inference", output / "result.json") from error

    monkeypatch.setattr(cli, "run_step_inference", fail)

    exit_code = cli.main(
        [
            "--step",
            str(step),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Traceback" in captured.err
    assert "result.json" in captured.err


@pytest.mark.parametrize(
    "case",
    [
        "missing-file",
        "wrong-suffix",
        "threshold",
        "device",
        "missing-step-argument",
    ],
)
def test_cli_rejects_invalid_arguments_and_writes_error_json(tmp_path, case):
    step = tmp_path / "part.step"
    checkpoint = tmp_path / "model.ckpt"
    output = tmp_path / "argument-error-output"
    step.write_bytes(b"STEP")
    checkpoint.write_bytes(b"CHECKPOINT")
    argv = [
        "--step",
        str(step),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output),
    ]
    if case == "missing-file":
        argv[1] = str(tmp_path / "missing.step")
    elif case == "wrong-suffix":
        argv[1] = __file__
    elif case == "threshold":
        argv.extend(["--affinity-threshold", "2"])
    elif case == "device":
        argv.extend(["--device", "tpu"])
    elif case == "missing-step-argument":
        argv = argv[2:]

    with pytest.raises(SystemExit) as caught:
        cli.main(argv)

    assert caught.value.code == 2
    document = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert document["status"] == "error"
    assert document["stage"] == "arguments"


def test_cli_help_builds_without_loading_occ():
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0
