"""Load Isaac-GR00T reference modules directly by file path, bypassing the
package init chain (which has a transformers-5.x-compat dataclass error that's
unrelated to what we're testing).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

GR00T_SRC = Path("/Users/rychmc/repos/robotics/Isaac-GR00T")


def _load_file_as_module(path: Path, name: str, extra_sys_modules: dict[str, ModuleType] | None = None) -> ModuleType:
    """Load `path` as a module named `name`, without executing any package __init__."""
    if extra_sys_modules:
        for k, v in extra_sys_modules.items():
            sys.modules[k] = v

    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None, f"could not load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_gr00t_dit() -> ModuleType:
    """Return the `gr00t/model/modules/dit.py` module (PyTorch reference)."""
    return _load_file_as_module(
        GR00T_SRC / "gr00t" / "model" / "modules" / "dit.py",
        name="_gr00t_ref_dit",
    )


def load_gr00t_embodiment_mlp() -> ModuleType:
    """Return `gr00t/model/modules/embodiment_conditioned_mlp.py` (PyTorch reference)."""
    return _load_file_as_module(
        GR00T_SRC / "gr00t" / "model" / "modules" / "embodiment_conditioned_mlp.py",
        name="_gr00t_ref_embodiment_mlp",
    )


def gr00t_src_available() -> bool:
    return GR00T_SRC.is_dir()
