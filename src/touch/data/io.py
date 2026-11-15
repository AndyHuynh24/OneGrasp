"""Small, safe file readers shared by the dataset loaders."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickler that refuses every global except numpy's (no arbitrary code on load)."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy" or module.startswith("numpy."):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load global {module}.{name}")


def load_numpy_pickle(path: str | Path) -> Any:
    """Load a pickle that may only contain builtins and numpy objects."""
    with open(path, "rb") as f:
        return _NumpyOnlyUnpickler(f).load()


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)
