"""Reassemble one parent's semantic predictions from overlapping debug crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def aggregate_probabilities(parent_faces, num_classes, crops):
    sums = np.zeros((parent_faces, num_classes), dtype=np.float64)
    counts = np.zeros(parent_faces, dtype=np.int32)
    for face_indices, probabilities in crops:
        indices = np.asarray(face_indices, dtype=np.int64)
        values = np.asarray(probabilities, dtype=np.float64)
        if indices.ndim != 1 or values.shape != (len(indices), num_classes):
            raise ValueError("crop predictions have inconsistent shapes")
        if np.any(indices < 0) or np.any(indices >= parent_faces):
            raise ValueError("crop contains an invalid parent face index")
        if not np.isfinite(values).all():
            raise ValueError("crop predictions contain non-finite values")
        np.add.at(sums, indices, values)
        np.add.at(counts, indices, 1)
    if np.any(counts == 0):
        raise ValueError("uncovered parent faces: {}".format(int((counts == 0).sum())))
    return sums / counts[:, None], counts


def aggregate_pair_logits(parent_faces, crops):
    sums = np.zeros((parent_faces, parent_faces), dtype=np.float64)
    counts = np.zeros((parent_faces, parent_faces), dtype=np.int32)
    for face_indices, logits in crops:
        indices = np.asarray(face_indices, dtype=np.int64)
        values = np.asarray(logits, dtype=np.float64)
        if values.shape != (len(indices), len(indices)):
            raise ValueError("crop relation logits have inconsistent shapes")
        if np.any(indices < 0) or np.any(indices >= parent_faces):
            raise ValueError("crop relation indices are outside parent graph")
        if not np.isfinite(values).all():
            raise ValueError("crop relation logits contain non-finite values")
        values = (values + values.T) / 2.0
        target = np.ix_(indices, indices)
        sums[target] += values
        counts[target] += 1
    observed = counts > 0
    average = np.divide(sums, counts, out=np.zeros_like(sums), where=observed)
    return average, observed


def evaluate(dataset_root, checkpoint, output):
    import torch
    from dgl.data.utils import load_graphs
    from data.dataset import CADSynth
    from models.brepseg_model import BrepSeg

    dataset_root = Path(dataset_root)
    checkpoint = Path(checkpoint)
    output = Path(output)
    manifest = json.loads(
        (dataset_root / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    codes = manifest["index_to_semantic_code"]
    parent_path = dataset_root / "bin" / "001_parent.bin"
    parents, parent_metadata = load_graphs(str(parent_path))
    parent = parents[0]
    truth = parent.ndata["f"].cpu().numpy().astype(np.int64)
    sample_set = CADSynth(
        dataset_root, split="test", num_class=len(codes),
        require_instance=True, require_face_pointcloud=True,
        require_parametric_surface=True, require_cylindrical_pos=True,
        bin_root=dataset_root,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BrepSeg.load_from_checkpoint(
        str(checkpoint), map_location="cpu"
    ).to(device).eval()
    crop_predictions = []
    crop_relations = []
    with torch.no_grad():
        for index, sample in enumerate(sample_set):
            batch = sample_set._collate([sample])
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            probabilities, affinity = model.predict_with_affinity(batch)
            probabilities = probabilities.detach().cpu().numpy()
            _, metadata = load_graphs(str(sample_set.file_paths[index]))
            parent_indices = metadata["parent_face_index"].cpu().numpy()
            crop_predictions.append((parent_indices, probabilities))
            if affinity is not None:
                crop_relations.append((
                    parent_indices, affinity[0].detach().cpu().numpy(),
                ))

    average, votes = aggregate_probabilities(
        parent.num_nodes(), len(codes), crop_predictions,
    )
    predicted = average.argmax(axis=1)
    per_class = {}
    ious = []
    for class_index in sorted(set(truth.tolist()) | set(predicted.tolist())):
        actual = truth == class_index
        chosen = predicted == class_index
        true_positive = int(np.logical_and(actual, chosen).sum())
        false_positive = int(np.logical_and(~actual, chosen).sum())
        false_negative = int(np.logical_and(actual, ~chosen).sum())
        union = true_positive + false_positive + false_negative
        iou = true_positive / union if union else None
        if iou is not None:
            ious.append(iou)
        per_class[str(codes[class_index])] = {
            "reference_faces": int(actual.sum()),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "iou": iou,
        }
    document = {
        "purpose": "same-parent crop-aggregated memorization check; no held-out data",
        "dataset": str(dataset_root.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "parent_faces": parent.num_nodes(),
        "crop_count": len(sample_set),
        "face_accuracy": float((predicted == truth).mean()),
        "miou_union_classes": float(np.mean(ious)),
        "per_class": per_class,
        "parent_face_predictions": [
            {
                "face_index": index,
                "semantic_code": codes[int(predicted[index])],
                "reference_code": codes[int(truth[index])],
                "confidence": float(average[index, predicted[index]]),
                "crop_votes": int(votes[index]),
            }
            for index in range(parent.num_nodes())
        ],
    }
    if crop_relations:
        averaged_logits, observed = aggregate_pair_logits(
            parent.num_nodes(), crop_relations,
        )
        truth_relation = parent_metadata["instance"].cpu().numpy() > 0
        upper = np.triu(np.ones_like(observed, dtype=bool), k=1)
        compared = observed & upper
        predicted_relation = averaged_logits > 0
        tp = int(np.count_nonzero(compared & predicted_relation & truth_relation))
        fp = int(np.count_nonzero(compared & predicted_relation & ~truth_relation))
        fn = int(np.count_nonzero(compared & ~predicted_relation & truth_relation))
        denominator = 2 * tp + fp + fn
        document["raw_pairwise_relation"] = {
            "threshold_logit": 0.0,
            "observed_pairs": int(compared.sum()),
            "total_parent_pairs": int(upper.sum()),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "f1": (2 * tp / denominator) if denominator else None,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: value for key, value in document.items()
        if key != "parent_face_predictions"
    }, ensure_ascii=False, indent=2))
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evaluate(args.dataset, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
