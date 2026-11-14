"""The README's evidence block must be exactly what the script produces."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evidence", ROOT / "scripts" / "evidence.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rendered(evidence: ModuleType) -> str:
    # Rendering runs every simulation (~20 s), so do it once.
    block: str = evidence.render()
    return block


def test_readme_block_matches_evidence_script(evidence: ModuleType, rendered: str) -> None:
    assert evidence.readme_block() == rendered, "run `python scripts/evidence.py` and update README"


def test_check_flags_a_moved_number(
    evidence: ModuleType, rendered: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evidence, "render", lambda: rendered)
    assert evidence.main(["--check"]) == 0
    assert "0.1277" in rendered
    moved = rendered.replace("0.1277", "0.1278", 1)
    monkeypatch.setattr(evidence, "render", lambda: moved)
    assert evidence.main(["--check"]) == 1
