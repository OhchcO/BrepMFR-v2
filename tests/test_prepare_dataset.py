import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import prepare_dataset
from tools.prepare_dataset import (
    PreparationConfig,
    PreparationError,
    PreparationMismatchError,
    build_preparation_manifest,
    discover_raw_samples,
    ensure_prepared_dataset,
    resolve_model_splits,
    select_split_subsets,
)


def _label_document(sample_id: str) -> dict:
    classes = list(range(24))
    return {
        "seg": classes,
        "inst": [
            [int(row == column) for column in range(len(classes))]
            for row in range(len(classes))
        ],
    }


def make_raw_fixture(tmp_path: Path, sample_count: int) -> Path:
    raw_root = tmp_path / "raw"
    steps = raw_root / "steps"
    labels = raw_root / "labels"
    steps.mkdir(parents=True)
    labels.mkdir()
    for index in range(sample_count):
        sample_id = "sample_{0:03d}".format(index)
        (steps / (sample_id + ".step")).write_text(
            "STEP placeholder {0}\n".format(sample_id), encoding="utf-8"
        )
        (labels / (sample_id + ".json")).write_text(
            json.dumps(_label_document(sample_id)), encoding="utf-8"
        )
    return raw_root


def make_preparation_config(tmp_path: Path, sample_percent: float = 1.0) -> PreparationConfig:
    raw_root = make_raw_fixture(tmp_path, sample_count=30)
    return PreparationConfig(
        raw_root=raw_root,
        prepared_root=tmp_path / "prepared",
        extractor_file=tmp_path / "AAGExtractor.py",
        schema_file=tmp_path / "schema.json",
        sample_percent=sample_percent,
    )


def write_incompatible_manifest(config: PreparationConfig, sample_percent: float) -> None:
    config.prepared_root.mkdir(parents=True)
    incompatible = PreparationConfig(
        raw_root=config.raw_root,
        prepared_root=config.prepared_root,
        extractor_file=config.extractor_file,
        schema_file=config.schema_file,
        sample_percent=sample_percent,
    )
    selected = {"train": (), "val": (), "test": ()}
    manifest = build_preparation_manifest(incompatible, selected)
    manifest["successful_ids"] = {"train": [], "val": [], "test": []}
    (config.prepared_root / "preparation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_generated_split_is_model_level_disjoint_and_reproducible(tmp_path):
    raw = make_raw_fixture(tmp_path, sample_count=20)
    ids = tuple(sorted(discover_raw_samples(raw)))
    left = resolve_model_splits(raw, ids, (0.8, 0.1, 0.1), seed=2026)
    right = resolve_model_splits(raw, ids, (0.8, 0.1, 0.1), seed=2026)

    assert left == right
    assert not (set(left["train"]) & set(left["val"]))
    assert not (set(left["train"]) & set(left["test"]))
    assert not (set(left["val"]) & set(left["test"]))
    assert set().union(*map(set, left.values())) == set(ids)


def test_sample_percent_applies_to_train_and_val_but_not_test(tmp_path):
    raw = make_raw_fixture(tmp_path, sample_count=100)
    splits = resolve_model_splits(
        raw, tuple(sorted(discover_raw_samples(raw))), (0.8, 0.1, 0.1), 2026
    )
    selected = select_split_subsets(raw, splits, sample_percent=0.25, seed=2026)

    assert len(selected["train"]) < len(splits["train"])
    assert len(selected["val"]) < len(splits["val"])
    assert selected["test"] == splits["test"]


def test_partial_official_manifests_are_rejected(tmp_path):
    raw = make_raw_fixture(tmp_path, sample_count=10)
    (raw / "train.txt").write_text("sample_000\n", encoding="utf-8")

    with pytest.raises(ValueError, match="all three"):
        resolve_model_splits(
            raw, tuple(sorted(discover_raw_samples(raw))), (0.8, 0.1, 0.1), 2026
        )


def test_discovery_rejects_mismatched_step_and_label_stems(tmp_path):
    raw = make_raw_fixture(tmp_path, sample_count=2)
    (raw / "labels" / "sample_001.json").unlink()

    with pytest.raises(ValueError, match="stems"):
        discover_raw_samples(raw)


def test_cache_fingerprint_mismatch_fails_closed(tmp_path):
    config = make_preparation_config(tmp_path, sample_percent=0.01)
    write_incompatible_manifest(config, sample_percent=0.05)

    with pytest.raises(PreparationMismatchError, match="new prepared directory|rebuild"):
        ensure_prepared_dataset(config)


def test_complete_matching_cache_is_reused_without_loading_converter(tmp_path, monkeypatch):
    config = make_preparation_config(tmp_path)
    samples = discover_raw_samples(config.raw_root)
    splits = resolve_model_splits(
        config.raw_root, tuple(samples), config.split_ratios, config.seed
    )
    selected = select_split_subsets(
        config.raw_root, splits, config.sample_percent, config.seed
    )
    manifest = build_preparation_manifest(config, selected, samples=samples)
    manifest["successful_ids"] = {key: list(value) for key, value in selected.items()}
    config.prepared_root.mkdir()
    for split_name, sample_ids in selected.items():
        (config.prepared_root / (split_name + ".txt")).write_text(
            "\n".join(sample_ids) + "\n", encoding="utf-8"
        )
    for sample_id in samples:
        bin_file = config.prepared_root / "bin" / (sample_id + ".bin")
        bin_file.parent.mkdir(exist_ok=True)
        bin_file.write_bytes(b"prepared")
    (config.prepared_root / "preparation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: pytest.fail("converter must not load for a matching cache"),
    )

    prepared = ensure_prepared_dataset(config)

    assert prepared.reused is True
    assert prepared.train_ids == selected["train"]


def test_conversion_is_one_shot_and_records_successful_manifests(tmp_path, monkeypatch):
    config = make_preparation_config(tmp_path, sample_percent=1.0)
    calls = []

    def convert_selected_samples(sample_ids_by_split, paths, extractor_file, schema_file, options, overwrite):
        calls.append((sample_ids_by_split, paths, options, overwrite))
        for sample_ids in sample_ids_by_split.values():
            for sample_id in sample_ids:
                target = paths.bin_file(sample_id)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"prepared")
        return {
            "successful_ids": sample_ids_by_split,
            "failed_ids": [],
            "conversion": {"converted": sum(map(len, sample_ids_by_split.values()))},
        }

    class Paths:
        def __init__(self, dataset_root, subset_root, output_root):
            self.dataset_root = dataset_root
            self.subset_root = subset_root
            self.output_root = output_root

        def bin_file(self, sample_id):
            return self.output_root / "bin" / (sample_id + ".bin")

    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: SimpleNamespace(
            ConversionPaths=Paths, convert_selected_samples=convert_selected_samples
        ),
    )

    prepared = ensure_prepared_dataset(config)

    assert len(calls) == 1
    assert prepared.reused is False
    assert (config.prepared_root / "conversion_failures.json").read_text(encoding="utf-8") == "[]\n"
    assert (config.prepared_root / "preparation_manifest.json").is_file()


def test_conversion_failures_stop_preparation_and_are_persisted(tmp_path, monkeypatch):
    config = make_preparation_config(tmp_path)

    class Paths:
        def __init__(self, dataset_root, subset_root, output_root):
            self.output_root = output_root

        def bin_file(self, sample_id):
            return self.output_root / "bin" / (sample_id + ".bin")

    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: SimpleNamespace(
            ConversionPaths=Paths,
            convert_selected_samples=lambda *args: {
                "successful_ids": {"train": [], "val": [], "test": []},
                "failed_ids": [{"sample_id": "sample_000", "message": "bad STEP"}],
            },
        ),
    )

    with pytest.raises(PreparationError, match="failed"):
        ensure_prepared_dataset(config)

    failures = json.loads(
        (config.prepared_root / "conversion_failures.json").read_text(encoding="utf-8")
    )
    assert failures[0]["sample_id"] == "sample_000"
