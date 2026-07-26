"""Stable random-seed helpers shared by preparation and model entry points."""

import hashlib
import random
import warnings

import dgl
import numpy as np
import torch


_UINT32_MODULUS = 2**32


def derive_seed(base_seed: int, purpose: str, sample_id: str = "") -> int:
    """Derive a stable unsigned 32-bit seed for one stochastic purpose."""
    payload = "{}\0{}\0{}".format(int(base_seed), purpose, sample_id).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def seed_everything(base_seed: int, deterministic: bool = True) -> None:
    """Seed every project RNG and configure PyTorch determinism when requested."""
    seed = int(base_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    _configure_deterministic_algorithms(deterministic)


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's DataLoader-assigned worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % _UINT32_MODULUS
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(base_seed: int, purpose: str) -> torch.Generator:
    """Return a generator deterministically scoped to a DataLoader purpose."""
    generator = torch.Generator()
    generator.manual_seed(derive_seed(base_seed, purpose))
    return generator


def _configure_deterministic_algorithms(deterministic: bool) -> None:
    """Use PyTorch's deterministic operator guard when the installed version has it."""
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)
    except AttributeError:
        if deterministic:
            warnings.warn(
                "This PyTorch version cannot guarantee deterministic operators "
                "because torch.use_deterministic_algorithms is unavailable.",
                RuntimeWarning,
                stacklevel=2,
            )
