from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture(scope="session")
def _workspace_tmp_root(request: pytest.FixtureRequest) -> Path:
    """Session temp root anchored to repository root."""
    root_path = Path(str(request.config.rootpath))
    workspace_tmp_root = root_path / "temp_pytest" / "workspace_tmp"
    workspace_tmp_root.mkdir(parents=True, exist_ok=True)
    yield workspace_tmp_root
    shutil.rmtree(workspace_tmp_root, ignore_errors=True)


@pytest.fixture
def tmp_path(
    request: pytest.FixtureRequest,
    _workspace_tmp_root: Path,
) -> Path:
    """Workspace-local tmp path to avoid OS temp permission issues."""
    test_tmp_path = _workspace_tmp_root / f"{request.node.name}_{uuid4().hex}"
    test_tmp_path.mkdir(parents=True, exist_ok=True)
    return test_tmp_path
