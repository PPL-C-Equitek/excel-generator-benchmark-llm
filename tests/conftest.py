from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest


_TMP_ROOT = Path.cwd() / "temp_pytest" / "workspace_tmp"


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    """Workspace-local tmp path to avoid OS temp permission issues."""
    test_tmp_path = _TMP_ROOT / f"{request.node.name}_{uuid4().hex}"
    test_tmp_path.mkdir(parents=True, exist_ok=True)
    return test_tmp_path
