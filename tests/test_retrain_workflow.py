"""Checks of the weekly retrain workflow. GitHub cannot run here, so the file is read as text."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
from pathlib import Path

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "retrain.yml"
PROMOTE = Path(__file__).parent.parent / ".github" / "workflows" / "promote.yml"


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
    # The step of the main branch is not --no-mlflow: that run must be logged and registered.
    assert "--no-mlflow" not in step(workflow, "Train, test and register the model")


def step(workflow: str, name: str) -> str:
    """The lines of one step of the workflow, from its name up to the next step, without comments."""
    text = workflow[workflow.index(f"- name: {name}") :]
    end = text.find("- name:", 1)  # -1 if it is the last step
    text = text if end == -1 else text[:end]
    # Comment lines are left out: the comment ABOVE the next step would otherwise count
    # as part of this step (it names --no-mlflow, for example).
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


def test_only_the_main_branch_logs_and_registers():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    # A run on another branch must never register a model or move the alias (found in a
    # review on 18.11.2026): the registering step runs only on main ...
    register = step(workflow, "Train, test and register the model")
    assert "if: github.ref == 'refs/heads/main'" in register
    # ... and the step for every other branch trains with --no-mlflow and gets no token.
    other = step(workflow, "Train and test only")
    assert "if: github.ref != 'refs/heads/main'" in other
    assert "train.py --no-mlflow" in other
    assert "MLFLOW_TRACKING_PASSWORD" not in other


def test_training_stops_if_the_feature_table_is_old():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    # The freshness check runs after the table is pulled and before the training, so an
    # old table is never trained on or registered (found in a review on 15.11.2026).
    # .index() gives the position of the text in the file: the steps run top to bottom.
    pull = workflow.index("dvc pull data/features.dvc")
    check = workflow.index("python -m lakegeneva.features.check_fresh")
    training = workflow.index("python src/lakegeneva/training/train.py")
    assert pull < check < training


def test_the_token_comes_only_from_the_secret():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    # The token is handed over as an environment variable from GitHub's secrets, never
    # written into the file itself.
    assert "MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}" in workflow
    assert workflow.count("MLFLOW_TRACKING_PASSWORD:") == 1
    # The job only reads the repository: retraining never commits anything.
    assert "contents: read" in workflow


def test_a_promotion_by_hand_waits_for_a_running_retrain():
    retrain = WORKFLOW.read_text(encoding="utf-8")
    promote = PROMOTE.read_text(encoding="utf-8")
    # Both workflows use the same concurrency group, so GitHub runs them one after the
    # other: a retrain can never overwrite a promotion made while it runs (found in a
    # review on 20.11.2026).
    assert 'group: retrain' in retrain and 'group: retrain' in promote
    # Started by hand only, with the version as an input; only from the main branch.
    assert "workflow_dispatch:" in promote and "schedule:" not in promote
    assert "if: github.ref == 'refs/heads/main'" in promote
    # The typed version reaches the command through an environment variable, never
    # written into the command itself (no script injection) ...
    run = step(promote, "Move the alias champion to the version")
    assert "VERSION: ${{ inputs.version }}" in run
    assert 'track.py --promote "$VERSION"' in run
    assert "${{ inputs.version }}" not in next(line for line in run.splitlines() if "run:" in line)
    # ... and the token comes only from the secret.
    assert "MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}" in promote
