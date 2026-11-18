"""Render two verified MFInstSeg semantic comparison cases from real predictions.

The figure uses the seed-42 1% prepared BIN and checkpoint for BrepMFR,
the saved ``predicted_instances.json`` for BrepMFR++, and raw MFInstSeg
labels and STEP geometry. It deliberately includes both a gain and a loss.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from brepmfr.inference.pipeline import _load_checkpoint  # noqa: E402
from data.collator import collator  # noqa: E402
from data.dataset import CADSynth  # noqa: E402
from tools.mfrinstseg_to_bin import load_extractor, load_schema  # noqa: E402


DATA = Path(r"D:\Code\Datasets\MFRInstSeg")
BIN_DIR = Path(
    r"D:\Code\Datasets\MFRInstSeg_BrepMFR_ParamSurface_v4_shared_seed42\bin"
)
RESULTS = REPO / "results" / "paper_core_experiments"
BASE_DIR = RESULTS / "MFRInstSeg_1pct_uv_aag_seed42"
FULL_DIR = RESULTS / (
    "MFRInstSeg_1pct_spe_instance_positive_param_surface_intrinsic_"
    "untrimmed_gated_seed42"
)
OUT_DIR = Path(__file__).resolve().parent
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

COLORS = {17: "#2377b4", 22: "#ea8a2f"}
OTHER = "#cbd3d9"
ERROR = "#bb2525"
CASES = (
    ("20221121_154647_1569", "improvement", [(13, 17, 22), (14, 17, 22), (15, 17, 22), (16, 17, 22)]),
    ("20221121_154647_0", "limitation", [(6, 22, 17), (8, 22, 17), (15, 22, 17)]),
)


def one_file(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} in {folder}, found {len(matches)}")
    return matches[0]


def labels_for(sample_id: str) -> np.ndarray:
    path = DATA / "labels" / f"{sample_id}.json"
    entry = json.loads(path.read_text(encoding="utf-8"))[0]
    if entry[0] != sample_id:
        raise ValueError(f"Label ID mismatch: {sample_id}")
    seg = entry[1]["seg"]
    return np.asarray([int(seg[str(i)]) for i in range(len(seg))], dtype=np.int64)


def full_predictions(path: Path, sample_ids: set[str], labels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    records = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for record in records:
        sample_id = record["sample_id"]
        if sample_id not in sample_ids:
            continue
        prediction = np.full(len(labels[sample_id]), 24, dtype=np.int64)
        for instance in record["instances"]:
            prediction[instance["face_indices"]] = int(instance["semantic_class"])
        result[sample_id] = prediction
    if set(result) != sample_ids:
        raise ValueError(f"Missing full-method predictions: {sample_ids - set(result)}")
    return result


def make_dataset(loaded) -> CADSynth:
    dataset = CADSynth.__new__(CADSynth)
    dataset.split = "test"
    dataset.num_class = loaded.num_classes
    dataset.random_rotate = False
    dataset.require_instance = False
    dataset.require_face_pointcloud = loaded.requirements.require_face_pointcloud
    dataset.require_parametric_surface = loaded.requirements.require_parametric_surface
    dataset.require_cylindrical_pos = loaded.requirements.require_cylindrical_pos
    dataset.allow_unlabeled = False
    dataset.seed = loaded.seed
    return dataset


def baseline_predictions(sample_ids: set[str]) -> dict[str, np.ndarray]:
    checkpoint = one_file(BASE_DIR, "best-mIoU*.ckpt")
    loaded = _load_checkpoint(checkpoint)
    loaded.model.eval()
    dataset = make_dataset(loaded)
    predictions = {}
    for sample_id in sorted(sample_ids):
        bin_path = BIN_DIR / f"{sample_id}.bin"
        dataset.file_paths = [bin_path]
        batch = collator(
            [dataset.load_one_graph(bin_path)],
            multi_hop_max_dist=16,
            spatial_pos_max=32,
        )
        with torch.inference_mode():
            prediction = loaded.model(batch).argmax(dim=-1).cpu().numpy()
        predictions[sample_id] = prediction.astype(np.int64)
    return predictions


def face_meshes(sample_id: str) -> list[np.ndarray]:
    extractor_class, _ = load_extractor(REPO / "tools" / "AAGExtractor.py")
    extractor = extractor_class(
        DATA / "steps" / f"{sample_id}.step",
        load_schema(REPO / "tools" / "schema.json"),
    )
    extractor.process()
    mesher = BRepMesh_IncrementalMesh(extractor.body, 0.025, False, 0.25, True)
    mesher.Perform()
    meshes = []
    for face_index in range(len(extractor._graph.nodes)):
        face = extractor._graph.nodes[face_index]["face"].topods_shape()
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        if triangulation is None or triangulation.NbTriangles() == 0:
            raise RuntimeError(f"No triangulation for face {face_index} of {sample_id}")
        vertices = []
        for i in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(i).Transformed(location.Transformation())
            vertices.append((point.X(), point.Y(), point.Z()))
        vertices = np.asarray(vertices, dtype=float)
        triangles = []
        for i in range(1, triangulation.NbTriangles() + 1):
            a, b, c = triangulation.Triangle(i).Get()
            triangles.append(vertices[[a - 1, b - 1, c - 1]])
        meshes.append(np.asarray(triangles, dtype=float))
    return meshes


def render_panel(ax, meshes: list[np.ndarray], classes: np.ndarray, truth: np.ndarray, view: tuple[int, int]) -> None:
    polygons = []
    facecolors = []
    edgecolors = []
    for face_index, triangles in enumerate(meshes):
        color = COLORS.get(int(classes[face_index]), OTHER)
        is_error = classes[face_index] != truth[face_index]
        polygons.extend(triangles)
        facecolors.extend([to_rgba(color)] * len(triangles))
        edge = to_rgba(ERROR, 0.85) if is_error else to_rgba("#56616c", 0.08)
        edgecolors.extend([edge] * len(triangles))
    collection = Poly3DCollection(
        polygons,
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidths=0.3,
        zsort="average",
    )
    ax.add_collection3d(collection)
    points = np.concatenate([mesh.reshape(-1, 3) for mesh in meshes], axis=0)
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    radius = max((points.max(axis=0) - points.min(axis=0)).max() / 2, 0.1) * 1.08
    ax.set(xlim=(center[0] - radius, center[0] + radius),
           ylim=(center[1] - radius, center[1] + radius),
           zlim=(center[2] - radius, center[2] + radius))
    ax.set_box_aspect((1, 1, 1), zoom=1.42)
    ax.set_proj_type("ortho")
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()


def main() -> None:
    torch.set_num_threads(4)
    sample_ids = {case[0] for case in CASES}
    labels = {sample_id: labels_for(sample_id) for sample_id in sample_ids}
    base = baseline_predictions(sample_ids)
    full = full_predictions(
        one_file(FULL_DIR, "predicted_instances.json"), sample_ids, labels
    )
    for sample_id, kind, expected in CASES:
        truth = labels[sample_id]
        if len(base[sample_id]) != len(truth) or len(full[sample_id]) != len(truth):
            raise ValueError(f"Face count mismatch for {sample_id}")
        baseline_errors = [
            (i, int(truth[i]), int(base[sample_id][i]))
            for i in range(len(truth)) if base[sample_id][i] != truth[i]
        ]
        full_errors = [
            (i, int(truth[i]), int(full[sample_id][i]))
            for i in range(len(truth)) if full[sample_id][i] != truth[i]
        ]
        if (kind == "improvement" and (baseline_errors != expected or full_errors)) or (
            kind == "limitation" and (baseline_errors or full_errors != expected)
        ):
            raise RuntimeError(f"Predictions changed for {sample_id}: {baseline_errors}, {full_errors}")

        meshes = face_meshes(sample_id)
        if len(meshes) != len(truth):
            raise ValueError(f"STEP and labels have different face counts for {sample_id}")
        fig = plt.figure(figsize=(12.5, 7.0), facecolor="white")
        views = ((23, -58), (20, -10)) if kind == "limitation" else (
            (23, -58), (20, 122)
        )
        panels = (("真实语义", truth), ("BrepMFR（UV-AAG）", base[sample_id]),
                  ("BrepMFR++（本文）", full[sample_id]))
        for row, view in enumerate(views):
            for col, (name, classes) in enumerate(panels):
                ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="3d")
                render_panel(ax, meshes, classes, truth, view)
                if row == 0:
                    ax.set_title(name, fontsize=14, pad=5, weight="semibold")
        description = (
            "改进案例：BrepMFR 将面 13–16 的第 17 类误判为第 22 类；本文方法四面均判对。"
            if kind == "improvement" else
            "局限案例：BrepMFR 全部判对；本文方法将面 6、8、15 的第 22 类误判为第 17 类。"
        )
        fig.suptitle(f"MFInstSeg｜1% 训练集｜seed 42｜零件 {sample_id}",
                     fontsize=15, y=0.985)
        fig.legend(
            handles=[Patch(facecolor=COLORS[17], label="17 矩形盲槽"),
                     Patch(facecolor=COLORS[22], label="22 矩形盲台阶"),
                     Patch(facecolor=OTHER, label="其他类别"),
                     Patch(facecolor="white", edgecolor=ERROR, label="误分类面（红色边线）")],
            loc="lower center", bbox_to_anchor=(0.5, 0.058), ncol=4, frameon=False,
            fontsize=10.5,
        )
        fig.text(0.5, 0.022, description, ha="center", fontsize=10.5)
        fig.text(0.018, 0.70, "视角一", rotation=90, ha="center", va="center", fontsize=10)
        fig.text(0.018, 0.33, "视角二", rotation=90, ha="center", va="center", fontsize=10)
        fig.subplots_adjust(left=0.045, right=0.99, top=0.90, bottom=0.13,
                            wspace=0.0, hspace=-0.08)
        output = OUT_DIR / f"MFInstSeg_seed42_{kind}_comparison.png"
        fig.savefig(output, dpi=300, facecolor="white")
        fig.savefig(output.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)
        print(f"{output} | faces={len(truth)} | baseline_errors={baseline_errors} | full_errors={full_errors}")


if __name__ == "__main__":
    main()
