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


RAW_STEP = Path(r"D:\Code\Datasets\MFRInstSeg\steps\20221121_154647_0.step")
RAW_LABEL = Path(r"D:\Code\Datasets\MFRInstSeg\labels\20221121_154647_0.json")
PREPARED_BIN = Path(
    r"D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_shared_seed42"
    r"\bin\20221121_154647_0.bin"
)
CHECKPOINT = Path(
    r"D:\Code\BrepMFR\results\paper_core_experiments"
    r"\MFRInstSeg_1pct_spe_instance_positive_param_surface_intrinsic_"
    r"untrimmed_gated_seed42\0821\092832"
    r"\best-mIoU-187-0.939778.ckpt"
)
GOLDEN_PATH = (
    Path(__file__).resolve().parent / "golden" / "20221121_154647_0.json"
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


def test_step_inference_matches_prepared_bin_and_committed_golden(
    tmp_path, monkeypatch
):
    missing = [
        path
        for path in (RAW_STEP, PREPARED_BIN, CHECKPOINT)
        if not path.is_file()
    ]
    if missing:
        pytest.skip("external golden assets are unavailable: {0}".format(missing))
    assert RAW_LABEL.is_file(), "posthoc label asset is missing"
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
        "face_count": int(reference_semantic.shape[0]),
        "checkpoint_filename": CHECKPOINT.name,
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pytorch_lightning": pytorch_lightning.__version__,
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
    assert (
        golden["semantic_probability_sha256"]
        == record["semantic_probability_sha256"]
    )
    assert (
        golden["affinity_probability_sha256"]
        == record["affinity_probability_sha256"]
    )
    assert_prediction_records_close(actual.document["faces"], golden["faces"])
    assert_prediction_records_close(
        actual.document["instances"], golden["instances"]
    )
    for key, expected in golden["posthoc_metrics"].items():
        actual_metric = posthoc_metrics[key]
        if isinstance(expected, int):
            assert actual_metric == expected
        else:
            assert actual_metric == pytest.approx(expected, abs=1e-8)
