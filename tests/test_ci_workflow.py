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


def test_the_image_runs_the_pipeline_on_saved_answers_without_internet():
    # MS2 review point A3: not only --help, but the whole pipeline in the image, on the
    # saved answers of tests/fixtures/replay, with the network of the container off.
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step = workflow[workflow.index("- name: Run the pipeline in the image on saved API answers") :]
    assert "--network none" in step
    assert "-e LAKEGENEVA_REPLAY=/replay" in step
    assert '-v "$PWD/tests/fixtures/replay:/replay:ro"' in step
    # The table it built must name the issue day of the saved answers.
    assert 'grep -q "issue day D: 2026-11-26" replay.log' in step
    # And it runs after the image is built, in the image job.
    assert workflow.index("tags: lake-geneva-features:ci") < workflow.index("Run the pipeline in the image")


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
