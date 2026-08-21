import json
from dataclasses import dataclass
import os
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_prepare_cli_defaults_to_repository_aag_assets(tmp_path):
    args = prepare_dataset.build_parser().parse_args(
        ["--raw-dataset", str(tmp_path), "--prepared-dataset", str(tmp_path / "out")]
    )

    assert args.extractor_file == PROJECT_ROOT / "tools" / "AAGExtractor.py"
    assert args.schema_file == PROJECT_ROOT / "tools" / "schema.json"


def test_prepare_cli_accepts_shared_cache_and_donor_roots(tmp_path):
    shared = tmp_path / "shared"
    donor_one = tmp_path / "one_pct"
    donor_five = tmp_path / "five_pct"

    args = prepare_dataset.build_parser().parse_args(
        [
            "--raw-dataset",
            str(tmp_path),
            "--prepared-dataset",
            str(tmp_path / "ten_pct"),
            "--shared-bin-cache",
            str(shared),
            "--donor-prepared-dataset",
            str(donor_one),
            str(donor_five),
        ]
    )

    assert args.shared_bin_cache == shared
    assert args.donor_prepared_dataset == [donor_one, donor_five]


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
    def __init__(self, dataset_root, subset_root, output_root, bin_cache_root=None):
        self.dataset_root = dataset_root
        self.subset_root = subset_root
        self.output_root = output_root
        self.bin_cache_root = bin_cache_root

    def bin_file(self, sample_id):
        return (self.bin_cache_root or self.output_root) / "bin" / (sample_id + ".bin")


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


def test_manifest_initializes_effective_selection_and_exclusion_audit(tmp_path):
    config = make_preparation_config(tmp_path)
    samples = discover_raw_samples(config.raw_root)
    splits = resolve_model_splits(
        config.raw_root, tuple(samples), config.split_ratios, config.seed
    )
    selected = select_split_subsets(
        config.raw_root, splits, config.sample_percent, config.seed
    )

    manifest = build_preparation_manifest(config, selected, samples=samples)

    assert manifest["effective_selected_ids"] == {
        name: list(selected[name]) for name in ("train", "val", "test")
    }
    assert manifest["conversion_exclusions"] == []
    assert manifest["replacement_policy_version"] == 1


def test_replacement_outputs_do_not_change_request_fingerprint(tmp_path):
    config = make_preparation_config(tmp_path)
    samples = discover_raw_samples(config.raw_root)
    splits = resolve_model_splits(
        config.raw_root, tuple(samples), config.split_ratios, config.seed
    )
    selected = select_split_subsets(
        config.raw_root, splits, config.sample_percent, config.seed
    )
    manifest = build_preparation_manifest(config, selected, samples=samples)
    recovered = json.loads(json.dumps(manifest))
    recovered["effective_selected_ids"]["train"] = ["replacement"]
    recovered["conversion_exclusions"] = [{"failed_sample_id": "failed"}]

    assert prepare_dataset._request_fingerprint(manifest) == (
        prepare_dataset._request_fingerprint(recovered)
    )


def test_effective_selection_falls_back_to_legacy_selected_ids():
    legacy = {
        "selected_ids": {
            "train": ["train-id"],
            "val": ["val-id"],
            "test": ["test-id"],
        }
    }

    assert prepare_dataset._effective_selected_ids(legacy) == {
        "train": ("train-id",),
        "val": ("val-id",),
        "test": ("test-id",),
    }


def test_matching_cache_validates_and_returns_effective_selection(
    tmp_path, monkeypatch
):
    config = make_preparation_config(tmp_path, sample_percent=0.5)
    samples = discover_raw_samples(config.raw_root)
    splits = resolve_model_splits(
        config.raw_root, tuple(samples), config.split_ratios, config.seed
    )
    selected = select_split_subsets(
        config.raw_root, splits, config.sample_percent, config.seed
    )
    replacement = next(
        sample_id
        for sample_id in splits["train"]
        if sample_id not in selected["train"]
    )
    effective = {
        "train": tuple(sorted(selected["train"][:-1] + (replacement,))),
        "val": tuple(selected["val"]),
        "test": tuple(selected["test"]),
    }
    manifest = build_preparation_manifest(config, selected, samples=samples)
    manifest["effective_selected_ids"] = {
        name: list(effective[name]) for name in ("train", "val", "test")
    }
    manifest["successful_ids"] = {
        name: list(effective[name]) for name in ("train", "val", "test")
    }
    config.prepared_root.mkdir()
    for split_name, sample_ids in effective.items():
        (config.prepared_root / (split_name + ".txt")).write_text(
            "\n".join(sample_ids) + "\n", encoding="utf-8"
        )
        for sample_id in sample_ids:
            write_valid_contract_bin(
                config.prepared_root / "bin" / (sample_id + ".bin")
            )
    (config.prepared_root / "preparation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: fake_converter(
            lambda *args: pytest.fail("complete effective cache must be reused")
        ),
    )

    prepared = ensure_prepared_dataset(config)

    assert prepared.reused is True
    assert prepared.train_ids == effective["train"]
    assert prepared.val_ids == effective["val"]
    assert prepared.test_ids == effective["test"]


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

    def convert_selected_samples(
        sample_ids_by_split,
        paths,
        extractor_file,
        schema_file,
        options,
        overwrite,
        num_workers=None,
    ):
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


def test_shared_bin_identity_excludes_percentage_and_selected_ids(tmp_path):
    one = make_preparation_config(tmp_path, sample_percent=0.01)
    samples = discover_raw_samples(one.raw_root)
    splits = resolve_model_splits(one.raw_root, tuple(samples), one.split_ratios, one.seed)
    one_selected = select_split_subsets(one.raw_root, splits, 0.01, one.seed)
    ten = PreparationConfig(**{**one.__dict__, "sample_percent": 0.10})
    ten_selected = select_split_subsets(ten.raw_root, splits, 0.10, ten.seed)

    one_manifest = build_preparation_manifest(one, one_selected, samples=samples)
    ten_manifest = build_preparation_manifest(ten, ten_selected, samples=samples)

    assert prepare_dataset._bin_cache_fingerprint(one_manifest) == (
        prepare_dataset._bin_cache_fingerprint(ten_manifest)
    )


def test_shared_bin_identity_tracks_every_raw_source(tmp_path):
    config = make_preparation_config(tmp_path, sample_percent=0.01)
    samples = discover_raw_samples(config.raw_root)
    splits = resolve_model_splits(
        config.raw_root, tuple(samples), config.split_ratios, config.seed
    )
    selected = select_split_subsets(
        config.raw_root, splits, config.sample_percent, config.seed
    )

    manifest = build_preparation_manifest(config, selected, samples=samples)

    assert set(manifest["source_files"]) == set(samples)


def test_shared_cache_upgrades_compatible_legacy_source_subset(tmp_path):
    config = make_preparation_config(tmp_path, sample_percent=0.10)
    samples = discover_raw_samples(config.raw_root)
    splits = resolve_model_splits(
        config.raw_root, tuple(samples), config.split_ratios, config.seed
    )
    selected = select_split_subsets(
        config.raw_root, splits, config.sample_percent, config.seed
    )
    request = build_preparation_manifest(config, selected, samples=samples)
    selected_ids = {
        sample_id for split_ids in selected.values() for sample_id in split_ids
    }
    legacy_request = json.loads(json.dumps(request))
    legacy_request["source_files"] = {
        sample_id: source
        for sample_id, source in legacy_request["source_files"].items()
        if sample_id in selected_ids
    }
    shared = tmp_path / "shared"
    shared.mkdir()
    manifest_path = shared / "bin_cache_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "bin_cache_format_version": prepare_dataset.BIN_CACHE_FORMAT_VERSION,
                "fingerprint": prepare_dataset._bin_cache_fingerprint(
                    legacy_request
                ),
            }
        ),
        encoding="utf-8",
    )

    prepare_dataset._ensure_shared_cache_manifest(shared, request)

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "bin_cache_format_version": prepare_dataset.BIN_CACHE_FORMAT_VERSION,
        "fingerprint": prepare_dataset._bin_cache_fingerprint(request),
    }


def test_shared_cache_rejects_legacy_source_hash_mismatch(tmp_path):
    config = make_preparation_config(tmp_path, sample_percent=0.10)
    samples = discover_raw_samples(config.raw_root)
    splits = resolve_model_splits(
        config.raw_root, tuple(samples), config.split_ratios, config.seed
    )
    selected = select_split_subsets(
        config.raw_root, splits, config.sample_percent, config.seed
    )
    request = build_preparation_manifest(config, selected, samples=samples)
    selected_ids = {
        sample_id for split_ids in selected.values() for sample_id in split_ids
    }
    legacy_request = json.loads(json.dumps(request))
    legacy_request["source_files"] = {
        sample_id: source
        for sample_id, source in legacy_request["source_files"].items()
        if sample_id in selected_ids
    }
    first_source = next(iter(legacy_request["source_files"].values()))
    first_source["step"]["sha256"] = "0" * 64
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "bin_cache_manifest.json").write_text(
        json.dumps(
            {
                "bin_cache_format_version": prepare_dataset.BIN_CACHE_FORMAT_VERSION,
                "fingerprint": prepare_dataset._bin_cache_fingerprint(
                    legacy_request
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        PreparationMismatchError, match="Shared BIN cache fingerprint"
    ):
        prepare_dataset._ensure_shared_cache_manifest(shared, request)


def test_shared_cache_fingerprint_mismatch_fails_closed(tmp_path):
    config = make_preparation_config(tmp_path, sample_percent=0.10)
    samples = discover_raw_samples(config.raw_root)
    splits = resolve_model_splits(
        config.raw_root, tuple(samples), config.split_ratios, config.seed
    )
    selected = select_split_subsets(
        config.raw_root, splits, config.sample_percent, config.seed
    )
    request = build_preparation_manifest(config, selected, samples=samples)
    shared = tmp_path / "shared"
    prepare_dataset._ensure_shared_cache_manifest(shared, request)
    changed = json.loads(json.dumps(request))
    changed["conversion_options"]["max_point_count"] = 64

    with pytest.raises(PreparationMismatchError, match="Shared BIN cache fingerprint"):
        prepare_dataset._ensure_shared_cache_manifest(shared, changed)


def test_shared_cache_routes_bins_away_from_percentage_root(tmp_path, monkeypatch):
    base = make_preparation_config(tmp_path, sample_percent=1.0)
    shared = tmp_path / "shared"
    config = PreparationConfig(
        **{**base.__dict__, "shared_bin_cache_root": shared}
    )

    def convert_selected_samples(
        sample_ids_by_split, paths, *unused, num_workers=None
    ):
        assert paths.output_root == config.prepared_root
        assert paths.bin_cache_root == shared
        for sample_ids in sample_ids_by_split.values():
            for sample_id in sample_ids:
                write_valid_contract_bin(paths.bin_file(sample_id))
        return {"successful_ids": sample_ids_by_split, "failed_ids": []}

    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: fake_converter(convert_selected_samples),
    )

    prepared = ensure_prepared_dataset(config)

    assert prepared.root == config.prepared_root
    assert prepared.bin_root == shared
    assert (shared / "bin_cache_manifest.json").is_file()
    assert not (config.prepared_root / "bin").exists()


def test_compatible_donor_bins_are_hard_linked_into_shared_cache(tmp_path, monkeypatch):
    donor_config = make_preparation_config(tmp_path, sample_percent=1.0)
    samples = discover_raw_samples(donor_config.raw_root)
    splits = resolve_model_splits(
        donor_config.raw_root,
        tuple(samples),
        donor_config.split_ratios,
        donor_config.seed,
    )
    selected = select_split_subsets(
        donor_config.raw_root, splits, donor_config.sample_percent, donor_config.seed
    )
    donor_manifest = build_preparation_manifest(
        donor_config, selected, samples=samples
    )
    donor_manifest["successful_ids"] = {
        name: list(ids) for name, ids in selected.items()
    }
    donor_config.prepared_root.mkdir()
    for split_name, ids in selected.items():
        _write_manifest = donor_config.prepared_root / (split_name + ".txt")
        _write_manifest.write_text("\n".join(ids) + "\n", encoding="utf-8")
        for sample_id in ids:
            write_valid_contract_bin(
                donor_config.prepared_root / "bin" / (sample_id + ".bin")
            )
    (donor_config.prepared_root / "preparation_manifest.json").write_text(
        json.dumps(donor_manifest), encoding="utf-8"
    )
    shared = tmp_path / "shared"
    target_root = tmp_path / "ten_pct"
    config = PreparationConfig(
        **{
            **donor_config.__dict__,
            "prepared_root": target_root,
            "shared_bin_cache_root": shared,
            "donor_prepared_roots": (donor_config.prepared_root,),
        }
    )

    def convert_selected_samples(
        sample_ids_by_split, paths, *unused, num_workers=None
    ):
        for ids in sample_ids_by_split.values():
            for sample_id in ids:
                assert paths.bin_file(sample_id).is_file()
        return {"successful_ids": sample_ids_by_split, "failed_ids": []}

    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: fake_converter(convert_selected_samples),
    )

    ensure_prepared_dataset(config)

    sample_id = selected["train"][0]
    donor_bin = donor_config.prepared_root / "bin" / (sample_id + ".bin")
    shared_bin = shared / "bin" / (sample_id + ".bin")
    assert os.path.samefile(donor_bin, shared_bin)


def test_converter_receives_published_options_and_automatic_path_contract(tmp_path, monkeypatch):
    config = make_preparation_config(tmp_path)

    def convert_selected_samples(
        sample_ids_by_split,
        paths,
        extractor_file,
        schema_file,
        options,
        overwrite,
        num_workers=None,
    ):
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

    def convert_selected_samples(
        sample_ids_by_split, paths, *unused, num_workers=None
    ):
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

    def convert_selected_samples(
        sample_ids_by_split, paths, *unused, num_workers=None
    ):
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
            lambda *args, num_workers=None: {
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


def _replacement_test_selection(config):
    samples = discover_raw_samples(config.raw_root)
    splits = resolve_model_splits(
        config.raw_root, tuple(samples), config.split_ratios, config.seed
    )
    selected = select_split_subsets(
        config.raw_root, splits, config.sample_percent, config.seed
    )
    return splits, selected


def test_train_conversion_failure_is_replaced_without_changing_subset_size(
    tmp_path, monkeypatch
):
    config = make_preparation_config(tmp_path, sample_percent=0.5)
    splits, selected = _replacement_test_selection(config)
    failed_id = selected["train"][-1]
    calls = []

    def convert_selected_samples(
        sample_ids_by_split,
        paths,
        extractor_file,
        schema_file,
        options,
        overwrite,
        num_workers=None,
    ):
        batch = {
            name: tuple(sample_ids_by_split.get(name, ()))
            for name in ("train", "val", "test")
        }
        calls.append(batch)
        if len(calls) == 1:
            successful = {
                name: tuple(
                    sample_id
                    for sample_id in batch[name]
                    if sample_id != failed_id
                )
                for name in ("train", "val", "test")
            }
            for sample_ids in successful.values():
                for sample_id in sample_ids:
                    write_valid_contract_bin(paths.bin_file(sample_id))
            return {
                "successful_ids": successful,
                "failed_ids": [
                    {
                        "split": "train",
                        "sample_id": failed_id,
                        "exception_type": "ValueError",
                        "message": "Face 14 has no valid UV candidates",
                    }
                ],
                "conversion": {
                    "converted": sum(map(len, successful.values())),
                    "skipped": 0,
                },
            }
        replacement_ids = [sample_id for ids in batch.values() for sample_id in ids]
        assert len(replacement_ids) == 1
        write_valid_contract_bin(paths.bin_file(replacement_ids[0]))
        return {
            "successful_ids": batch,
            "failed_ids": [],
            "conversion": {"converted": 1, "skipped": 0},
        }

    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: fake_converter(convert_selected_samples),
    )

    prepared = ensure_prepared_dataset(config)

    assert len(calls) == 2
    replacement_id = calls[1]["train"][0]
    assert replacement_id in splits["train"]
    assert replacement_id not in selected["train"]
    assert len(prepared.train_ids) == len(selected["train"])
    assert failed_id not in prepared.train_ids
    assert replacement_id in prepared.train_ids
    assert set(selected["train"][:-1]).issubset(prepared.train_ids)
    assert prepared.test_ids == selected["test"]
    exclusions = json.loads(
        (config.prepared_root / "conversion_exclusions.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(exclusions) == 1
    assert exclusions[0]["failed_sample_id"] == failed_id
    assert exclusions[0]["replacement_sample_id"] == replacement_id
    assert exclusions[0]["replacement_succeeded"] is True
    assert json.loads(
        (config.prepared_root / "conversion_failures.json").read_text(
            encoding="utf-8"
        )
    ) == []
    for split_name, expected_ids in (
        ("train", prepared.train_ids),
        ("val", prepared.val_ids),
        ("test", prepared.test_ids),
    ):
        assert tuple(
            (config.prepared_root / (split_name + ".txt"))
            .read_text(encoding="utf-8")
            .splitlines()
        ) == expected_ids


def test_failed_replacement_is_audited_before_trying_another_candidate(
    tmp_path, monkeypatch
):
    config = make_preparation_config(tmp_path, sample_percent=0.5)
    _, selected = _replacement_test_selection(config)
    initial_failure = selected["train"][-1]
    calls = []

    def convert_selected_samples(
        sample_ids_by_split, paths, *unused, num_workers=None
    ):
        batch = {
            name: tuple(sample_ids_by_split.get(name, ()))
            for name in ("train", "val", "test")
        }
        calls.append(batch)
        if len(calls) == 1:
            successful = {
                name: tuple(
                    sample_id
                    for sample_id in batch[name]
                    if sample_id != initial_failure
                )
                for name in ("train", "val", "test")
            }
            for sample_ids in successful.values():
                for sample_id in sample_ids:
                    write_valid_contract_bin(paths.bin_file(sample_id))
            return {
                "successful_ids": successful,
                "failed_ids": [
                    {
                        "split": "train",
                        "sample_id": initial_failure,
                        "exception_type": "ValueError",
                        "message": "initial failure",
                    }
                ],
                "conversion": {"converted": sum(map(len, successful.values()))},
            }
        replacement_id = batch["train"][0]
        if len(calls) == 2:
            return {
                "successful_ids": {"train": [], "val": [], "test": []},
                "failed_ids": [
                    {
                        "split": "train",
                        "sample_id": replacement_id,
                        "exception_type": "ValueError",
                        "message": "replacement failure",
                    }
                ],
                "conversion": {"converted": 0},
            }
        write_valid_contract_bin(paths.bin_file(replacement_id))
        return {
            "successful_ids": batch,
            "failed_ids": [],
            "conversion": {"converted": 1},
        }

    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: fake_converter(convert_selected_samples),
    )

    prepared = ensure_prepared_dataset(config)

    assert len(calls) == 3
    first_replacement = calls[1]["train"][0]
    second_replacement = calls[2]["train"][0]
    assert first_replacement != second_replacement
    assert first_replacement not in prepared.train_ids
    assert second_replacement in prepared.train_ids
    exclusions = json.loads(
        (config.prepared_root / "conversion_exclusions.json").read_text(
            encoding="utf-8"
        )
    )
    assert [record["failed_sample_id"] for record in exclusions] == [
        initial_failure,
        first_replacement,
    ]
    assert [record["replacement_sample_id"] for record in exclusions] == [
        first_replacement,
        second_replacement,
    ]
    assert [record["replacement_succeeded"] for record in exclusions] == [
        False,
        True,
    ]


@pytest.mark.parametrize(
    "failure_kind",
    ("test", "missing_sample", "unknown_split", "unknown_selected_id"),
)
def test_nonreplaceable_conversion_failures_remain_fatal(
    tmp_path, monkeypatch, failure_kind
):
    config = make_preparation_config(tmp_path, sample_percent=0.5)
    _, selected = _replacement_test_selection(config)
    if failure_kind == "test":
        split_name = "test"
        failed_id = selected["test"][0]
    elif failure_kind == "missing_sample":
        split_name = "train"
        failed_id = None
    elif failure_kind == "unknown_split":
        split_name = "holdout"
        failed_id = selected["train"][0]
    else:
        split_name = "train"
        failed_id = "not_in_the_selected_subset"

    def convert_selected_samples(
        sample_ids_by_split, paths, *unused, num_workers=None
    ):
        successful = {
            name: tuple(
                sample_id
                for sample_id in sample_ids_by_split[name]
                if not (name == split_name and sample_id == failed_id)
            )
            for name in ("train", "val", "test")
        }
        return {
            "successful_ids": successful,
            "failed_ids": [
                {
                    "split": split_name,
                    "sample_id": failed_id,
                    "exception_type": "ValueError",
                    "message": failure_kind,
                }
            ],
        }

    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: fake_converter(convert_selected_samples),
    )

    with pytest.raises(PreparationError, match="failed"):
        ensure_prepared_dataset(config)

    persisted = json.loads(
        (config.prepared_root / "conversion_failures.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted[0]["message"] == failure_kind
    assert json.loads(
        (config.prepared_root / "conversion_exclusions.json").read_text(
            encoding="utf-8"
        )
    ) == []


def test_replacement_candidate_exhaustion_remains_fatal(tmp_path, monkeypatch):
    config = make_preparation_config(tmp_path, sample_percent=1.0)
    _, selected = _replacement_test_selection(config)
    failed_id = selected["train"][-1]

    def convert_selected_samples(
        sample_ids_by_split, paths, *unused, num_workers=None
    ):
        successful = {
            name: tuple(
                sample_id
                for sample_id in sample_ids_by_split[name]
                if sample_id != failed_id
            )
            for name in ("train", "val", "test")
        }
        for sample_ids in successful.values():
            for sample_id in sample_ids:
                write_valid_contract_bin(paths.bin_file(sample_id))
        return {
            "successful_ids": successful,
            "failed_ids": [
                {
                    "split": "train",
                    "sample_id": failed_id,
                    "exception_type": "ValueError",
                    "message": "no usable UV samples",
                }
            ],
        }

    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: fake_converter(convert_selected_samples),
    )

    with pytest.raises(PreparationError, match="failed"):
        ensure_prepared_dataset(config)

    persisted = json.loads(
        (config.prepared_root / "conversion_failures.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted[0]["sample_id"] == failed_id
    assert persisted[0]["reason"] == "replacement_candidates_exhausted"


def test_matching_failed_cache_converts_only_one_balanced_replacement(
    tmp_path, monkeypatch
):
    config = make_preparation_config(tmp_path, sample_percent=0.5)
    samples = discover_raw_samples(config.raw_root)
    splits, selected = _replacement_test_selection(config)
    failed_id = selected["train"][-1]
    successful = {
        "train": list(selected["train"][:-1]),
        "val": list(selected["val"]),
        "test": list(selected["test"]),
    }
    failure = {
        "split": "train",
        "sample_id": failed_id,
        "exception_type": "ValueError",
        "message": "Face 14 has no valid UV candidates",
    }
    manifest = build_preparation_manifest(config, selected, samples=samples)
    manifest["successful_ids"] = successful
    manifest["failed_ids"] = [failure]
    manifest["conversion"] = {
        "converted": sum(len(ids) for ids in successful.values()),
        "skipped": 0,
    }
    config.prepared_root.mkdir()
    for split_name, sample_ids in successful.items():
        (config.prepared_root / (split_name + ".txt")).write_text(
            "\n".join(sample_ids) + ("\n" if sample_ids else ""),
            encoding="utf-8",
        )
        for sample_id in sample_ids:
            write_valid_contract_bin(
                config.prepared_root / "bin" / (sample_id + ".bin")
            )
    (config.prepared_root / "preparation_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (config.prepared_root / "conversion_failures.json").write_text(
        json.dumps([failure]), encoding="utf-8"
    )
    calls = []

    def convert_selected_samples(
        sample_ids_by_split, paths, *unused, num_workers=None
    ):
        batch = {
            name: tuple(sample_ids_by_split.get(name, ()))
            for name in ("train", "val", "test")
        }
        calls.append(batch)
        submitted = [sample_id for ids in batch.values() for sample_id in ids]
        assert len(submitted) == 1
        replacement_id = submitted[0]
        assert replacement_id not in set().union(*map(set, successful.values()))
        assert replacement_id not in set().union(*map(set, selected.values()))
        assert replacement_id in splits["train"]
        write_valid_contract_bin(paths.bin_file(replacement_id))
        return {
            "successful_ids": batch,
            "failed_ids": [],
            "conversion": {"converted": 1, "skipped": 0},
        }

    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: fake_converter(convert_selected_samples),
    )

    prepared = ensure_prepared_dataset(config)

    assert prepared.reused is False
    assert len(calls) == 1
    assert len(prepared.train_ids) == len(selected["train"])
    assert failed_id not in prepared.train_ids
    assert set(successful["train"]).issubset(prepared.train_ids)
    assert prepared.val_ids == tuple(selected["val"])
    assert prepared.test_ids == tuple(selected["test"])
    assert json.loads(
        (config.prepared_root / "conversion_failures.json").read_text(
            encoding="utf-8"
        )
    ) == []
    exclusions = json.loads(
        (config.prepared_root / "conversion_exclusions.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(exclusions) == 1
    assert exclusions[0]["failed_sample_id"] == failed_id


def test_unstructured_replacement_converter_error_is_fatal_without_retry(
    tmp_path, monkeypatch
):
    config = make_preparation_config(tmp_path, sample_percent=0.5)
    _, selected = _replacement_test_selection(config)
    failed_id = selected["train"][-1]
    calls = []

    def convert_selected_samples(
        sample_ids_by_split, paths, *unused, num_workers=None
    ):
        batch = {
            name: tuple(sample_ids_by_split.get(name, ()))
            for name in ("train", "val", "test")
        }
        calls.append(batch)
        if len(calls) > 1:
            raise RuntimeError("converter setup failed")
        successful = {
            name: tuple(
                sample_id for sample_id in batch[name] if sample_id != failed_id
            )
            for name in ("train", "val", "test")
        }
        for sample_ids in successful.values():
            for sample_id in sample_ids:
                write_valid_contract_bin(paths.bin_file(sample_id))
        return {
            "successful_ids": successful,
            "failed_ids": [
                {
                    "split": "train",
                    "sample_id": failed_id,
                    "exception_type": "ValueError",
                    "message": "initial model failure",
                }
            ],
        }

    monkeypatch.setattr(
        prepare_dataset,
        "_load_converter",
        lambda: fake_converter(convert_selected_samples),
    )

    with pytest.raises(PreparationError, match="failed"):
        ensure_prepared_dataset(config)

    assert len(calls) == 2
    persisted = json.loads(
        (config.prepared_root / "conversion_failures.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted == [
        {
            "exception_type": "RuntimeError",
            "message": "converter setup failed",
            "sample_id": None,
        }
    ]
