"""Load Isaac-GR00T reference modules directly by file path, bypassing the
package init chain (which has a transformers-5.x-compat dataclass error that's
unrelated to what we're testing).

Set the ``GR00T_REF_PATH`` environment variable to point at your local clone
of https://github.com/NVIDIA/Isaac-GR00T. If unset and the default path
below does not exist, parity tests skip gracefully with an explanatory message
rather than running green (which would hide real regressions).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

_DEFAULT_GR00T_SRC = Path.home() / "repos" / "robotics" / "Isaac-GR00T"
GR00T_SRC = Path(os.environ.get("GR00T_REF_PATH", str(_DEFAULT_GR00T_SRC)))


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


def skip_reason_if_unavailable() -> str | None:
    """Explicit skip message — parity tests that silently pass because they
    never ran are worse than missing tests; this makes the skip loud."""
    if GR00T_SRC.is_dir():
        return None
    return (
        f"Isaac-GR00T reference not found at {GR00T_SRC}. Set GR00T_REF_PATH "
        f"to a local clone of https://github.com/NVIDIA/Isaac-GR00T to run "
        f"parity tests."
    )
