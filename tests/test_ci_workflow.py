"""Tests for .github/workflows/ci.yml: CI checks the code with ruff before the tests."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
import tomllib  # reads pyproject.toml (standard library since Python 3.11)
from pathlib import Path

ROOT = Path(__file__).parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def test_ci_runs_ruff_before_the_tests():
    # MS2 review point E5: "add ruff to CI". ruff comes first, because it takes a second.
    workflow = WORKFLOW.read_text(encoding="utf-8")
    ruff = workflow.index("run: uv run ruff check")
    tests = workflow.index("run: uv run pytest")
    assert ruff < tests


def test_ruff_is_pinned_and_requires_type_hints_in_src():
    settings = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    # ruff is a dev dependency, so uv.lock pins its version and the image stays without it.
    assert any(package.startswith("ruff") for package in settings["dependency-groups"]["dev"])
    assert 'name = "ruff"' in (ROOT / "uv.lock").read_text(encoding="utf-8")
    lint = settings["tool"]["ruff"]["lint"]
    # The rules are named (a newer ruff with other defaults checks the same), and type
    # hints (ANN) are required everywhere except in the tests.
    assert "ANN" in lint["select"]
    assert lint["per-file-ignores"]["tests/*"] == ["ANN"]
