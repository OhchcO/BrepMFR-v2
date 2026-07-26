from contextlib import ExitStack
from unittest.mock import call, patch

import pytest

import utils.reproducibility as reproducibility
from utils.reproducibility import DEFAULT_SEED, derive_seed, make_torch_generator


def test_derive_seed_is_stable_and_purpose_specific():
    first = derive_seed(purpose="split", sample_id="model_001")
    assert DEFAULT_SEED == 2026
    assert first == 0xE038A013
    assert first == derive_seed(purpose="split", sample_id="model_001")
    assert 0 <= first < 2**32
    assert first != derive_seed(purpose="point_sampling", sample_id="model_001")


def test_derive_seed_includes_sample_id():
    assert derive_seed(purpose="split", sample_id="model_001") != derive_seed(
        purpose="split", sample_id="model_002"
    )


def test_torch_generator_replays_sequence():
    import torch

    left = torch.rand(4, generator=make_torch_generator(purpose="train_loader"))
    right = torch.rand(4, generator=make_torch_generator(purpose="train_loader"))
    assert torch.equal(left, right)


def test_torch_generator_uses_the_purpose_derived_seed():
    with patch.object(reproducibility.torch, "Generator") as generator_type:
        generator = generator_type.return_value

        assert make_torch_generator(purpose="train_loader") is generator

    generator.manual_seed.assert_called_once_with(
        derive_seed(DEFAULT_SEED, "train_loader")
    )


def test_seed_everything_seeds_all_process_rngs(monkeypatch):
    monkeypatch.setattr(reproducibility.torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(reproducibility.torch.backends.cudnn, "deterministic", False)
    with ExitStack() as stack:
        random_seed = stack.enter_context(
            patch.object(reproducibility.random, "seed")
        )
        numpy_seed = stack.enter_context(
            patch.object(reproducibility.np.random, "seed")
        )
        torch_seed = stack.enter_context(
            patch.object(reproducibility.torch, "manual_seed")
        )
        stack.enter_context(
            patch.object(reproducibility.torch.cuda, "is_available", return_value=True)
        )
        cuda_seed = stack.enter_context(
            patch.object(reproducibility.torch.cuda, "manual_seed_all")
        )
        dgl_seed = stack.enter_context(patch.object(reproducibility.dgl, "seed"))
        dgl_random_seed = stack.enter_context(
            patch.object(reproducibility.dgl.random, "seed")
        )
        configure = stack.enter_context(
            patch.object(reproducibility, "_configure_deterministic_algorithms")
        )
        reproducibility.seed_everything()

    for seed_call in (
        random_seed,
        numpy_seed,
        torch_seed,
        cuda_seed,
        dgl_seed,
        dgl_random_seed,
    ):
        seed_call.assert_called_once_with(DEFAULT_SEED)
    assert reproducibility.torch.backends.cudnn.benchmark is False
    assert reproducibility.torch.backends.cudnn.deterministic is True
    configure.assert_called_once_with(True)


def test_seed_worker_uses_the_pytorch_assigned_seed():
    worker_seed = 2**32 + 17
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(reproducibility.torch, "initial_seed", return_value=worker_seed)
        )
        random_seed = stack.enter_context(
            patch.object(reproducibility.random, "seed")
        )
        numpy_seed = stack.enter_context(
            patch.object(reproducibility.np.random, "seed")
        )
        reproducibility.seed_worker(worker_id=3)

    random_seed.assert_called_once_with(17)
    numpy_seed.assert_called_once_with(17)


def test_older_pytorch_warns_when_warn_only_is_unsupported():
    with patch.object(
        reproducibility.torch,
        "use_deterministic_algorithms",
        side_effect=[TypeError, None],
    ) as configure:
        with pytest.warns(RuntimeWarning, match="warn_only deterministic algorithms"):
            reproducibility._configure_deterministic_algorithms(True)

    assert configure.call_args_list == [call(True, warn_only=True), call(True)]
