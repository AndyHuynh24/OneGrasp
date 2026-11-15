"""Where the chumpy-free MANO pickles (MANO_RIGHT.pkl / MANO_LEFT.pkl) live: $MANO_DIR, else
data/mano in the repo (git-ignored), else ~/.cache/fah/mano (the old default; ~/.cache was wiped
twice on 2026-11-13, so the repo copy comes first). MANO's licence forbids redistribution."""

from __future__ import annotations

import os
from pathlib import Path


def mano_dir() -> Path:
    if os.environ.get("MANO_DIR"):
        return Path(os.environ["MANO_DIR"])
    repo = Path(__file__).resolve().parents[2] / "data/mano"
    return repo if (repo / "MANO_RIGHT.pkl").exists() else Path.home() / ".cache/fah/mano"


MANO_DIR = mano_dir()
