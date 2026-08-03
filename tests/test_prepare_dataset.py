import json
from dataclasses import dataclass
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


def make_raw_fixture(tmp_path: Path, sample_count: int, suffix: str = ".step") -> Path:
    raw_root = tmp_path / "raw"
    steps = raw_root / "steps"
    labels = raw_root / "labels"
    steps.mkdir(parents=True)
    labels.mkdir()
    for index in range(sample_count):
        sample_id = "sample_{0:03d}".format(index)
        (steps / (sample_id + suffix)).write_text(
            "STEP placeholder {0}\n".format(sample_id), encoding="utf-8"
        )
        (labels / (sample_id + ".json")).write_text(
            json.dumps(_label_document(sample_id)), encoding="utf-8"
        )
    return raw_root


def make_preparation_config(tmp_path: Path, sample_percent: float = 1.0) -> PreparationConfig:
    raw_root = make_raw_fixture(tmp_path, sample_count=30)
    extractor_file = tmp_path / "AAGExtractor.py"
    schema_file = tmp_path / "schema.json"
    extractor_file.write_text("extractor-v1\n", encoding="utf-8")
    schema_file.write_text('{"schema": 1}\n', encoding="utf-8")
    return PreparationConfig(
        raw_root=raw_root,
        prepared_root=tmp_path / "prepared",
        extractor_file=extractor_file,
        schema_file=schema_file,
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
    splits = resolve_model_splits(
        config.raw_root, tuple(discover_raw_samples(config.raw_root)), config.split_ratios, config.seed
    )
    selected = select_split_subsets(
        config.raw_root, splits, incompatible.sample_percent, incompatible.seed
    )
    manifest = build_preparation_manifest(incompatible, selected)
    manifest["successful_ids"] = {"train": [], "val": [], "test": []}
    (config.prepared_root / "preparation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


@dataclass(frozen=True)
class FakePointSamplingConfig:
    mode: str
    fixed_count: int
    min_count: int
    max_count: int
    area_exponent: float
    candidate_grids: tuple


@dataclass(frozen=True)
class FakeConversionOptions:
    seed: int
    point_sampling: FakePointSamplingConfig
    include_face_pointcloud: bool
    include_cylindrical_pos: bool
    include_parametric_surface: bool


@dataclass(frozen=True)
class FakeBinRequirements:
    require_instance: bool
    require_pointcloud: bool
    require_cylindrical: bool
    require_parametric_surface: bool


class FakePaths:
    def __init__(self, dataset_root, subset_root, output_root):
        self.dataset_root = dataset_root
        self.subset_root = subset_root
        self.output_root = output_root

    def bin_file(self, sample_id):
        return self.output_root / "bin" / (sample_id + ".bin")


def write_valid_contract_bin(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "face_points": [2, 96, 10],
                "face_point_mask": [2, 96],
                "face_point_count": [2],
                "face_surface_metric": [2, 96, 3],
                "face_trim_neighbors": [2, 96, 16],
                "cylindrical_pos": [2, 4],
            }
        ),
        encoding="utf-8",
    )


def contract_validator(
    path: Path, expected_nodes=None, requirements=None, point_sampling=None
) -> None:
    assert expected_nodes is None
    assert requirements == FakeBinRequirements(True, True, True, True)
    assert isinstance(point_sampling, FakePointSamplingConfig)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == {
        "face_points": [2, 96, 10],
        "face_point_mask": [2, 96],
        "face_point_count": [2],
        "face_surface_metric": [2, 96, 3],
        "face_trim_neighbors": [2, 96, 16],
        "cylindrical_pos": [2, 4],
    }


def fake_converter(convert_selected_samples, validator=contract_validator):
    return SimpleNamespace(
        ConversionPaths=FakePaths,
        PointSamplingConfig=FakePointSamplingConfig,
        ConversionOptions=FakeConversionOptions,
        BinRequirements=FakeBinRequirements,
        validate_bin=validator,
        convert_selected_samples=convert_selected_samples,
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


def test_manifest_publishes_10d_point_feature_contract(tmp_path):
    config = make_preparation_config(tmp_path)
    samples = discover_raw_samples(config.raw_root)
    splits = resolve_model_splits(
        config.raw_root, tuple(samples), config.split_ratios, config.seed
    )
    selected = select_split_subsets(
        config.raw_root, splits, config.sample_percent, config.seed
    )

    manifest = build_preparation_manifest(config, selected, samples=samples)

    assert manifest["converter_format_version"] == 4
    assert manifest["point_feature_contract"] == {
        "dimension": 10,
        "order": [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "u_normalized",
            "v_normalized",
            "mean_curvature_times_length",
            "gaussian_curvature_times_length_squared",
        ],
        "uv_range": [-1.0, 1.0],
        "characteristic_length": "sampled_full_part_bounding_box_diagonal",
    }


def test_point_feature_contract_participates_in_cache_fingerprint(tmp_path):
    config = make_preparation_config(tmp_path)
    samples = discover_raw_samples(config.raw_root)
    splits = resolve_model_splits(
        config.raw_root, tuple(samples), config.split_ratios, config.seed
    )
    selected = select_split_subsets(
        config.raw_root, splits, config.sample_percent, config.seed
    )
    manifest = build_preparation_manifest(config, selected, samples=samples)
    changed = json.loads(json.dumps(manifest))
    changed["point_feature_contract"]["dimension"] = 11

    assert prepare_dataset._request_fingerprint(
        manifest
    ) != prepare_dataset._request_fingerprint(changed)


def test_partial_official_manifests_are_rejected(tmp_path):
    raw = make_raw_fixture(tmp_path, sample_count=10)
    (raw / "train.txt").write_text("sample_000\n", encoding="utf-8")

    with pytest.raises(ValueError, match="all three"):
        resolve_model_splits(
            raw, tuple(sorted(discover_raw_samples(raw))), (0.8, 0.1, 0.1), 2026
        )


def test_complete_official_manifests_preserve_membership_and_order(tmp_path):
    raw = make_raw_fixture(tmp_path, sample_count=5)
    expected = {
        "train": ("sample_003", "sample_000", "sample_004"),
        "val": ("sample_002",),
        "test": ("sample_001",),
    }
    for split_name, sample_ids in expected.items():
        (raw / (split_name + ".txt")).write_text(
            "\n".join(sample_ids) + "\n", encoding="utf-8"
        )

    resolved = resolve_model_splits(raw, tuple(discover_raw_samples(raw)), (0.8, 0.1, 0.1), 99)
    selected = select_split_subsets(raw, resolved, sample_percent=1.0, seed=2026)

    assert resolved == expected
    assert selected == expected


def test_official_manifests_reject_unknown_membership(tmp_path):
    raw = make_raw_fixture(tmp_path, sample_count=2)
    manifests = {
        "train": "missing\n",
        "val": "sample_000\n",
        "test": "sample_001\n",
    }
    for split_name, text in manifests.items():
        (raw / (split_name + ".txt")).write_text(text, encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_model_splits(raw, tuple(discover_raw_samples(raw)), (0.8, 0.1, 0.1), 2026)


def test_discovery_accepts_case_insensitive_stp_and_rejects_duplicate_stems(tmp_path):
    raw = make_raw_fixture(tmp_path, sample_count=2)
    original = raw / "steps" / "sample_000.step"
    original.rename(original.with_suffix(".STP"))

    assert discover_raw_samples(raw)["sample_000"].step_file.suffix == ".STP"

    # Use a genuinely different supported suffix.  On Windows, ``.STP`` and
    # ``.stp`` name the same file and therefore cannot form a duplicate pair.
    (raw / "steps" / "sample_000.step").write_text(
        "duplicate", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"(?i)duplicate"):
        discover_raw_samples(raw)


@pytest.mark.parametrize(
    ("sample_count", "expected_counts"),
    [(1, (1, 0, 0)), (2, (2, 0, 0)), (3, (1, 1, 1)), (4, (2, 1, 1)), (10, (8, 1, 1))],
)
def test_generated_tiny_splits_are_deterministic_and_exhaustive(tmp_path, sample_count, expected_counts):
    raw = make_raw_fixture(tmp_path, sample_count=sample_count)
    splits = resolve_model_splits(raw, tuple(discover_raw_samples(raw)), (0.8, 0.1, 0.1), 2026)

    assert tuple(len(splits[name]) for name in ("train", "val", "test")) == expected_counts
    assert set().union(*map(set, splits.values())) == set(discover_raw_samples(raw))


def test_fractional_sparse_split_selects_available_classes(tmp_path):
    raw = make_raw_fixture(tmp_path, sample_count=4)
    sparse_label = {"seg": [0], "inst": [[1]]}
    for label_file in (raw / "labels").glob("*.json"):
        label_file.write_text(json.dumps(sparse_label), encoding="utf-8")
    splits = {
        "train": ("sample_000", "sample_001"),
        "val": ("sample_002",),
        "test": ("sample_003",),
    }

    selected = select_split_subsets(raw, splits, sample_percent=0.5, seed=2026)

    assert len(selected["train"]) == 1
    assert len(selected["val"]) == 1
    assert selected["test"] == splits["test"]


def test_subset_validation_rejects_blank_ids_and_invalid_percent(tmp_path):
    raw = make_raw_fixture(tmp_path, sample_count=3)
    with pytest.raises(ValueError, match="non-empty"):
        select_split_subsets(
            raw,
            {"train": ("sample_000", ""), "val": ("sample_001",), "test": ("sample_002",)},
            0.5,
            2026,
        )
    with pytest.raises(ValueError, match="sample_percent"):
        select_split_subsets(
            raw,
            {"train": ("sample_000",), "val": ("sample_001",), "test": ("sample_002",)},
            0.0,
            2026,
        )


@pytest.mark.parametrize("split_ratios", [(0.8, 0.2), (0.8, 0.1, 0.2), (0.8, 0.1, 0.0)])
def test_split_ratio_validation_rejects_invalid_contracts(tmp_path, split_ratios):
    raw = make_raw_fixture(tmp_path, sample_count=3)
    with pytest.raises(ValueError):
        resolve_model_splits(raw, tuple(discover_raw_samples(raw)), split_ratios, 2026)


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


@pytest.mark.parametrize("changed_file", ("extractor_file", "schema_file"))
def test_extractor_and_schema_content_changes_invalidate_cache(tmp_path, monkeypatch, changed_file):
    config = make_preparation_config(tmp_path)
    samples = discover_raw_samples(config.raw_root)
    splits = resolve_model_splits(config.raw_root, tuple(samples), config.split_ratios, config.seed)
    selected = select_split_subsets(config.raw_root, splits, config.sample_percent, config.seed)
    manifest = build_preparation_manifest(config, selected, samples=samples)
    manifest["successful_ids"] = {name: list(ids) for name, ids in selected.items()}
    config.prepared_root.mkdir()
    for split_name, sample_ids in selected.items():
        (config.prepared_root / (split_name + ".txt")).write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
        for sample_id in sample_ids:
            write_valid_contract_bin(config.prepared_root / "bin" / (sample_id + ".bin"))
    (config.prepared_root / "preparation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(prepare_dataset, "_load_converter", lambda: fake_converter(lambda *args: None))
    path = getattr(config, changed_file)
    path.write_text(path.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")

    with pytest.raises(PreparationMismatchError, match="fingerprint"):
        ensure_prepared_dataset(config)


def test_invalid_cached_bin_is_not_reused(tmp_path, monkeypatch):
    config = make_preparation_config(tmp_path)
    samples = discover_raw_samples(config.raw_root)
    splits = resolve_model_splits(config.raw_root, tuple(samples), config.split_ratios, config.seed)
    selected = select_split_subsets(config.raw_root, splits, config.sample_percent, config.seed)
    manifest = build_preparation_manifest(config, selected, samples=samples)
    manifest["successful_ids"] = {name: list(ids) for name, ids in selected.items()}
    config.prepared_root.mkdir()
    for split_name, sample_ids in selected.items():
        (config.prepared_root / (split_name + ".txt")).write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
        for sample_id in sample_ids:
            write_valid_contract_bin(config.prepared_root / "bin" / (sample_id + ".bin"))
    (config.prepared_root / "bin" / (selected["train"][0] + ".bin")).write_text("{}", encoding="utf-8")
    (config.prepared_root / "preparation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(prepare_dataset, "_load_converter", lambda: fake_converter(lambda *args: None))

    with pytest.raises(PreparationMismatchError, match="incomplete"):
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
        write_valid_contract_bin(bin_file)
    (config.prepared_root / "preparation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: fake_converter(lambda *args: pytest.fail("converter must not run for a matching cache")),
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
                write_valid_contract_bin(target)
        return {
            "successful_ids": sample_ids_by_split,
            "failed_ids": [],
            "conversion": {"converted": sum(map(len, sample_ids_by_split.values()))},
        }

    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: fake_converter(convert_selected_samples),
    )

    prepared = ensure_prepared_dataset(config)

    assert len(calls) == 1
    assert prepared.reused is False
    assert (config.prepared_root / "conversion_failures.json").read_text(encoding="utf-8") == "[]\n"
    assert (config.prepared_root / "preparation_manifest.json").is_file()


def test_converter_receives_published_options_and_automatic_path_contract(tmp_path, monkeypatch):
    config = make_preparation_config(tmp_path)

    def convert_selected_samples(sample_ids_by_split, paths, extractor_file, schema_file, options, overwrite):
        assert isinstance(options, FakeConversionOptions)
        assert options == FakeConversionOptions(
            seed=config.seed,
            point_sampling=FakePointSamplingConfig(
                mode=config.point_sampling_mode,
                fixed_count=config.fixed_point_count,
                min_count=config.min_point_count,
                max_count=config.max_point_count,
                area_exponent=config.area_sampling_exponent,
                candidate_grids=config.candidate_grids,
            ),
            include_face_pointcloud=True,
            include_cylindrical_pos=True,
            include_parametric_surface=True,
        )
        assert paths.dataset_root == config.raw_root
        assert paths.subset_root == config.prepared_root
        assert paths.output_root == config.prepared_root
        for sample_ids in sample_ids_by_split.values():
            for sample_id in sample_ids:
                write_valid_contract_bin(paths.bin_file(sample_id))
        return {"successful_ids": sample_ids_by_split, "failed_ids": []}

    monkeypatch.setattr(prepare_dataset, "_load_converter", lambda: fake_converter(convert_selected_samples))

    ensure_prepared_dataset(config)


@pytest.mark.parametrize("broken_manifest", ("not json", "[]", "{}"))
def test_rebuild_recovers_malformed_or_legacy_manifest_without_recursive_deletion(tmp_path, monkeypatch, broken_manifest):
    config = make_preparation_config(tmp_path)
    rebuilt = PreparationConfig(**{**config.__dict__, "rebuild": True})
    rebuilt.prepared_root.mkdir()
    sentinel = rebuilt.prepared_root / "unrelated" / "keep.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("preserve", encoding="utf-8")
    (rebuilt.prepared_root / "preparation_manifest.json").write_text(broken_manifest, encoding="utf-8")

    def convert_selected_samples(sample_ids_by_split, paths, *unused):
        for sample_ids in sample_ids_by_split.values():
            for sample_id in sample_ids:
                write_valid_contract_bin(paths.bin_file(sample_id))
        return {"successful_ids": sample_ids_by_split, "failed_ids": []}

    monkeypatch.setattr(prepare_dataset, "_load_converter", lambda: fake_converter(convert_selected_samples))

    prepared = ensure_prepared_dataset(rebuilt)

    assert prepared.reused is False
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_missing_converter_output_is_persisted_as_structured_failure(tmp_path, monkeypatch):
    config = make_preparation_config(tmp_path)

    def convert_selected_samples(sample_ids_by_split, paths, *unused):
        first = True
        for sample_ids in sample_ids_by_split.values():
            for sample_id in sample_ids:
                if first:
                    first = False
                    continue
                write_valid_contract_bin(paths.bin_file(sample_id))
        return {"successful_ids": sample_ids_by_split, "failed_ids": []}

    monkeypatch.setattr(prepare_dataset, "_load_converter", lambda: fake_converter(convert_selected_samples))

    with pytest.raises(PreparationError, match="failed"):
        ensure_prepared_dataset(config)

    failures = json.loads((config.prepared_root / "conversion_failures.json").read_text(encoding="utf-8"))
    assert failures[0]["sample_id"] is not None
    assert failures[0]["reason"] == "missing_or_invalid_bin"


def test_conversion_failures_stop_preparation_and_are_persisted(tmp_path, monkeypatch):
    config = make_preparation_config(tmp_path)

    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: fake_converter(
            lambda *args: {
                "successful_ids": {"train": [], "val": [], "test": []},
                "failed_ids": [{"sample_id": "sample_000", "message": "bad STEP"}],
            }
        ),
    )

    with pytest.raises(PreparationError, match="failed"):
        ensure_prepared_dataset(config)

    failures = json.loads(
        (config.prepared_root / "conversion_failures.json").read_text(encoding="utf-8")
    )
    assert failures[0]["sample_id"] == "sample_000"
