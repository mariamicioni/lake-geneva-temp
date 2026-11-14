"""Checks of the weekly retrain workflow. GitHub cannot run here, so the file is read as text."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
from pathlib import Path

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "retrain.yml"


def test_retraining_is_scheduled_weekly_and_can_be_started_by_hand():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    # The MS3 requirement: retraining starts on its own, not only by hand.
    assert 'cron: "0 6 * * 1"' in workflow  # every Monday, 06:00 UTC
    assert "workflow_dispatch:" in workflow


def test_retraining_reads_the_feature_store_and_runs_the_training_pipeline():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    # Only the feature table comes from the bucket, then the same script as on the laptop.
    assert "dvc pull data/features.dvc" in workflow
    assert "python src/lakegeneva/training/train.py" in workflow
    # Not with --no-mlflow: the run must be logged and registered.
    assert "--no-mlflow" not in workflow


def test_the_token_comes_only_from_the_secret():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    # The token is handed over as an environment variable from GitHub's secrets, never
    # written into the file itself.
    assert "MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}" in workflow
    assert workflow.count("MLFLOW_TRACKING_PASSWORD:") == 1
    # The job only reads the repository: retraining never commits anything.
    assert "contents: read" in workflow
