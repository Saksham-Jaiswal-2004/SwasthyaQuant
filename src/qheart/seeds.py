"""One place that seeds every RNG. Nothing else in the codebase calls seed().

Why this exists: a result you cannot reproduce is not a result. If two modules
seed independently, the same config can produce different numbers on different
days and you will not find out until you are defending the table.

Usage:
    from qheart.seeds import seed_everything, spawn
    seed_everything(cfg["seed"])            # once, at process start
    rng = spawn(cfg["seed"], "split", rep=3)  # per-purpose child generators
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

__all__ = ["seed_everything", "spawn", "derive"]


def derive(base: int, *tags: object) -> int:
    """Deterministically derive a child seed from a base seed and string tags.

    Stable across processes and Python versions -- unlike hash(), which is
    salted per process unless PYTHONHASHSEED is set.
    """
    key = "|".join([str(base), *(str(t) for t in tags)]).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def spawn(base: int, *tags: object) -> np.random.Generator:
    """A fresh, independent Generator for one purpose (splitting, init, shots...)."""
    return np.random.default_rng(derive(base, *tags))


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed stdlib, numpy, and -- if installed -- torch and pennylane.

    torch and pennylane are imported lazily so the classical-only install
    (requirements.txt) never needs them.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # Slower, but two runs of the same config must agree.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
