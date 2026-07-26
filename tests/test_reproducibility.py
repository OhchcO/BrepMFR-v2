from utils.reproducibility import derive_seed, make_torch_generator


def test_derive_seed_is_stable_and_purpose_specific():
    first = derive_seed(2026, "split", "model_001")
    assert first == derive_seed(2026, "split", "model_001")
    assert 0 <= first < 2**32
    assert first != derive_seed(2026, "point_sampling", "model_001")


def test_torch_generator_replays_sequence():
    import torch

    left = torch.rand(4, generator=make_torch_generator(2026, "train_loader"))
    right = torch.rand(4, generator=make_torch_generator(2026, "train_loader"))
    assert torch.equal(left, right)
