"""Every third-party import in the pipeline must be pinned in requirements-full.lock.

The weekly refresh workflow installs only that lock. This failed once in
production when yfinance left the app requirements and nothing else listed it.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_MODULES = ["screener.py", "evaluate.py", "generate_signals.py", "config.py"]
# import name -> distribution name where they differ
DIST_NAME = {"sklearn": "scikit-learn", "yaml": "pyyaml"}
LOCAL_MODULES = {"screener", "evaluate", "generate_signals", "config"}


def pinned(lock: Path) -> set[str]:
    return {m.group(1).lower() for m in re.finditer(r"^([A-Za-z0-9_.-]+)==", lock.read_text(), re.M)}


def top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("module", PIPELINE_MODULES)
def test_pipeline_imports_are_pinned_in_full_lock(module):
    locked = pinned(ROOT / "requirements-full.lock")
    stdlib = sys.stdlib_module_names
    missing = sorted(
        DIST_NAME.get(name, name).lower().replace("_", "-")
        for name in top_level_imports(ROOT / module)
        if name not in stdlib
        and name not in LOCAL_MODULES
        and DIST_NAME.get(name, name).lower().replace("_", "-") not in locked
    )
    assert not missing, f"{module} imports {missing}, not pinned in requirements-full.lock"


def test_app_imports_are_pinned_in_dev_lock():
    locked = pinned(ROOT / "requirements-dev.lock")
    stdlib = sys.stdlib_module_names
    missing = sorted(
        name
        for name in top_level_imports(ROOT / "app.py")
        if name not in stdlib and name not in LOCAL_MODULES and name.lower() not in locked
    )
    assert not missing, f"app.py imports {missing}, not pinned in requirements-dev.lock"
