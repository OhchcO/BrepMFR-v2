import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import pytest
import pytorch_lightning
import torch

from brepmfr.inference.cache import sha256_file
from brepmfr.inference.pipeline import (
    InferenceRequest,
    _load_checkpoint,
    run_step_inference,
)
from brepmfr.inference.postprocess import postprocess_predictions
from data.collator import collator
from data.dataset import CADSynth
from tools.mfrinstseg_to_bin import load_label
from tools.test_mfrinstseg import evaluate_instance_graph


TEST_ROOT = Path(__file__).resolve().parent
GOLDEN_PATH = TEST_ROOT / "golden" / "20221121_154647_0.json"
COMMITTED_STEP = (
    TEST_ROOT / "assets" / "step_inference" / "20221121_154647_0.step"
)
EXPECTED_STEP_SHA256 = (
    "00bc5acd0f49b1f11e113eebba656fce9003029d53b82b12ed586976d13e2bbd"
)


def _optional_external_path(variable):
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else None


RAW_STEP = COMMITTED_STEP
RAW_LABEL = _optional_external_path("BREPMFR_GOLDEN_LABEL")
PREPARED_BIN = _optional_external_path("BREPMFR_GOLDEN_PREPARED_BIN")
CHECKPOINT = _optional_external_path("BREPMFR_GOLDEN_CHECKPOINT")
_EXTERNAL_GOLDEN_ASSETS = {
    "BREPMFR_GOLDEN_LABEL": RAW_LABEL,
    "BREPMFR_GOLDEN_PREPARED_BIN": PREPARED_BIN,
    "BREPMFR_GOLDEN_CHECKPOINT": CHECKPOINT,
}
_RUNTIME_KEYS = (
    "platform",
    "machine",
    "python",
    "torch",
    "pytorch_lightning",
    "device",
)


def test_committed_golden_step_is_portable_and_content_addressed():
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    assert COMMITTED_STEP.is_file(), "committed golden STEP asset is missing"
    assert golden["source_split"] == "test"
    assert golden["step_filename"] == COMMITTED_STEP.name
    assert golden["step_sha256"] == EXPECTED_STEP_SHA256
    assert sha256_file(COMMITTED_STEP) == EXPECTED_STEP_SHA256


def _device_environment(device):
    if device.type != "cuda":
        return {
            "type": device.type,
            "name": None,
            "capability": None,
            "cuda": None,
            "cudnn": None,
        }
    return {
        "type": device.type,
        "name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def _runtime_matches(golden, record):
    return all(golden.get(key) == record.get(key) for key in _RUNTIME_KEYS)


def _assert_probability_hashes(golden, record):
    if not _runtime_matches(golden, record):
        return
    assert (
        golden["semantic_probability_sha256"]
        == record["semantic_probability_sha256"]
    )
    assert (
        golden["affinity_probability_sha256"]
        == record["affinity_probability_sha256"]
    )


def _labeled_reference_batch(bin_path, loaded, device):
    dataset = CADSynth.__new__(CADSynth)
    dataset.split = "test"
    dataset.num_class = loaded.num_classes
    dataset.random_rotate = False
    dataset.require_instance = True
    dataset.require_face_pointcloud = loaded.requirements.require_face_pointcloud
    dataset.require_parametric_surface = (
        loaded.requirements.require_parametric_surface
    )
    dataset.require_cylindrical_pos = loaded.requirements.require_cylindrical_pos
    dataset.allow_unlabeled = False
    dataset.seed = loaded.seed
    dataset.file_paths = [bin_path.resolve()]
    sample = dataset.load_one_graph(bin_path)
    batch = collator([sample], multi_hop_max_dist=16, spatial_pos_max=32)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _reference_predictions(device):
    loaded = _load_checkpoint(CHECKPOINT)
    batch = _labeled_reference_batch(PREPARED_BIN, loaded, device)
    model = loaded.model.to(device)
    model.eval()
    with torch.inference_mode():
        semantic, affinity_logits = model.predict_with_affinity(batch)
    assert affinity_logits is not None
    face_count = int((~batch["padding_mask"][0]).sum().item())
    semantic = semantic[:face_count].detach().cpu().numpy()
    affinity_logits = affinity_logits[0, :face_count, :face_count].detach().cpu().numpy()
    spatial_pos = batch["spatial_pos"][0, :face_count, :face_count].cpu().numpy()
    del model, loaded, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return semantic, affinity_logits, spatial_pos


def assert_prediction_records_close(actual, expected, atol=1e-5):
    assert len(actual) == len(expected)
    for actual_record, expected_record in zip(actual, expected):
        assert set(actual_record) == set(expected_record)
        for key in actual_record:
            if key == "confidence":
                assert actual_record[key] == pytest.approx(
                    expected_record[key], abs=atol
                )
            else:
                assert actual_record[key] == expected_record[key]


def _posthoc_metrics(reference_semantic, affinity_logits, spatial_pos, label):
    evaluation = evaluate_instance_graph(
        sample_id="20221121_154647_0",
        affinity_logits=affinity_logits,
        predicted_semantic=reference_semantic.argmax(axis=1),
        ground_truth_semantic=label.semantic,
        target_affinity=label.instance,
        spatial_pos=spatial_pos,
        threshold=0.5,
        background_class_id=24,
    )
    return {
        "per_face_accuracy": float(
            np.mean(reference_semantic.argmax(axis=1) == label.semantic)
        ),
        "clustering_pairwise_f1": float(
            evaluation["clustering"]["pairwise"]["f1"]
        ),
        "adjusted_rand_index": float(
            evaluation["clustering"]["adjusted_rand_index"]
        ),
        "predicted_instance_count": int(
            evaluation["clustering"]["predicted_instance_count"]
        ),
        "ground_truth_instance_count": int(
            evaluation["clustering"]["ground_truth_instance_count"]
        ),
    }


def test_probability_hashes_are_strict_only_for_matching_runtime():
    golden = {
        "python": "3.8.19",
        "torch": "1.13.1+cu116",
        "pytorch_lightning": "1.9.5",
        "device": {"type": "cuda", "name": "test-gpu"},
        "semantic_probability_sha256": "semantic-golden",
        "affinity_probability_sha256": "affinity-golden",
    }
    different_runtime = dict(golden)
    different_runtime["device"] = {"type": "cpu", "name": None}
    different_runtime["semantic_probability_sha256"] = "semantic-cpu"
    different_runtime["affinity_probability_sha256"] = "affinity-cpu"

    _assert_probability_hashes(golden, different_runtime)
    assert _runtime_matches(golden, different_runtime) is False

    matching_runtime = dict(golden)
    assert _runtime_matches(golden, matching_runtime) is True
    matching_runtime["semantic_probability_sha256"] = "changed"
    with pytest.raises(AssertionError):
        _assert_probability_hashes(golden, matching_runtime)


def test_step_inference_matches_prepared_bin_and_committed_golden(
    tmp_path, monkeypatch
):
    missing = [
        variable
        for variable, path in _EXTERNAL_GOLDEN_ASSETS.items()
        if path is None or not path.is_file()
    ]
    if missing:
        pytest.skip(
            "set existing paths for external golden assets: {0}".format(missing)
        )
    update_golden = os.environ.get("UPDATE_STEP_INFERENCE_GOLDEN") == "1"
    if not update_golden:
        assert GOLDEN_PATH.is_file(), "committed STEP inference golden is missing"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_semantic, reference_affinity_logits, reference_spatial_pos = (
        _reference_predictions(device)
    )
    reference_affinity = 1.0 / (
        1.0
        + np.exp(-np.clip(reference_affinity_logits, -60.0, 60.0))
    )
    reference_postprocess = postprocess_predictions(
        reference_semantic,
        reference_affinity_logits,
        reference_spatial_pos,
        threshold=0.5,
        background_class_id=24,
    )

    original_path_open = Path.open

    def guard_label_open(path, *args, **kwargs):
        if path.resolve() == RAW_LABEL.resolve():
            raise AssertionError("inference attempted to read the dataset label")
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard_label_open)
    output_dir = tmp_path / "recognition-output"
    request = InferenceRequest(
        step_path=RAW_STEP,
        checkpoint_path=CHECKPOINT,
        output_dir=output_dir,
        device=str(device),
    )
    actual = run_step_inference(request)
    repeated = run_step_inference(request)

    np.testing.assert_allclose(
        actual.semantic_probabilities,
        reference_semantic,
        rtol=0.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        actual.affinity_probabilities,
        reference_affinity,
        rtol=0.0,
        atol=1e-5,
    )
    assert actual.document["cache"]["hit"] is False
    assert repeated.document["cache"]["hit"] is True
    assert_prediction_records_close(
        repeated.document["faces"], actual.document["faces"]
    )
    assert_prediction_records_close(
        repeated.document["instances"], actual.document["instances"]
    )

    monkeypatch.setattr(Path, "open", original_path_open)
    label = load_label(RAW_LABEL)
    posthoc_metrics = _posthoc_metrics(
        reference_semantic,
        reference_affinity_logits,
        reference_spatial_pos,
        label,
    )
    record = {
        "schema_version": "1.0",
        "sample_id": "20221121_154647_0",
        "source_split": "test",
        "step_filename": RAW_STEP.name,
        "step_sha256": sha256_file(RAW_STEP),
        "face_count": int(reference_semantic.shape[0]),
        "checkpoint_filename": CHECKPOINT.name,
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pytorch_lightning": pytorch_lightning.__version__,
        "device": _device_environment(device),
        "semantic_probability_sha256": hashlib.sha256(
            np.ascontiguousarray(reference_semantic, dtype=np.float32).tobytes()
        ).hexdigest(),
        "affinity_probability_sha256": hashlib.sha256(
            np.ascontiguousarray(reference_affinity, dtype=np.float32).tobytes()
        ).hexdigest(),
        "faces": reference_postprocess.faces,
        "instances": reference_postprocess.instances,
        "posthoc_metrics": posthoc_metrics,
    }
    assert record["face_count"] == 41
    if update_golden:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if update_golden:
        assert golden == record
    assert golden["checkpoint_sha256"] == record["checkpoint_sha256"]
    _assert_probability_hashes(golden, record)
    record_atol = 1e-5 if _runtime_matches(golden, record) else 1e-3
    assert_prediction_records_close(
        actual.document["faces"], golden["faces"], atol=record_atol
    )
    assert_prediction_records_close(
        actual.document["instances"], golden["instances"], atol=record_atol
    )
    for key, expected in golden["posthoc_metrics"].items():
        actual_metric = posthoc_metrics[key]
        if isinstance(expected, int):
            assert actual_metric == expected
        else:
            assert actual_metric == pytest.approx(expected, abs=1e-8)
