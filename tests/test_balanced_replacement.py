import json

import pytest

from tools.sample_mfrinstseg_1pct import (
    select_balanced_replacement_id,
)


def make_label_root(tmp_path, rows):
    root = tmp_path / "dataset"
    labels = root / "labels"
    labels.mkdir(parents=True)
    for sample_id, instance_counts in rows.items():
        semantic = []
        instance_ids = []
        next_instance = 0
        for class_id, count in enumerate(instance_counts):
            for _ in range(count):
                semantic.append(class_id)
                instance_ids.append(next_instance)
                next_instance += 1
        affinity = [
            [int(left == right) for right in instance_ids]
            for left in instance_ids
        ]
        (labels / (sample_id + ".json")).write_text(
            json.dumps({"seg": semantic, "inst": affinity}),
            encoding="utf-8",
        )
    return root


def test_replacement_minimizes_balance_objective(tmp_path):
    kept = [1] * 24
    kept[1] = 0
    fills_missing = [0] * 24
    fills_missing[1] = 1
    adds_dominant = [0] * 24
    adds_dominant[0] = 2
    root = make_label_root(
        tmp_path,
        {
            "kept": kept,
            "fills_missing": fills_missing,
            "adds_dominant": adds_dominant,
        },
    )

    result = select_balanced_replacement_id(
        root,
        successful_ids=("kept",),
        candidate_ids=("fills_missing", "adds_dominant"),
        base_seed=42,
        split_name="train",
        attempt=1,
    )

    assert result.sample_id == "fills_missing"
    assert result.balance_before[0] == 1
    assert result.balance_after == (0, 0.0, 0)


def test_replacement_tie_break_is_reproducible(tmp_path):
    one_per_class = [1] * 24
    equal_candidate = [0] * 24
    equal_candidate[0] = 1
    root = make_label_root(
        tmp_path,
        {
            "kept": one_per_class,
            "candidate_a": equal_candidate,
            "candidate_b": equal_candidate,
        },
    )

    first = select_balanced_replacement_id(
        root,
        successful_ids=("kept",),
        candidate_ids=("candidate_a", "candidate_b"),
        base_seed=42,
        split_name="val",
        attempt=3,
    )
    second = select_balanced_replacement_id(
        root,
        successful_ids=("kept",),
        candidate_ids=("candidate_a", "candidate_b"),
        base_seed=42,
        split_name="val",
        attempt=3,
    )

    assert first == second
    assert first.sample_id in {"candidate_a", "candidate_b"}


@pytest.mark.parametrize(
    ("successful_ids", "candidate_ids", "split_name", "attempt", "message"),
    [
        (("kept",), (), "train", 1, "candidate_ids"),
        (("kept", "kept"), ("candidate",), "train", 1, "unique"),
        (("kept",), ("kept",), "train", 1, "disjoint"),
        (("kept",), ("candidate",), "test", 1, "split_name"),
        (("kept",), ("candidate",), "train", 0, "attempt"),
    ],
)
def test_replacement_rejects_invalid_selection_contract(
    tmp_path,
    successful_ids,
    candidate_ids,
    split_name,
    attempt,
    message,
):
    root = make_label_root(
        tmp_path,
        {
            "kept": [1] * 24,
            "candidate": [1] * 24,
        },
    )

    with pytest.raises(ValueError, match=message):
        select_balanced_replacement_id(
            root,
            successful_ids=successful_ids,
            candidate_ids=candidate_ids,
            base_seed=42,
            split_name=split_name,
            attempt=attempt,
        )
