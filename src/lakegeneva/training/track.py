"""Experiment tracking and model registry with MLflow (hosted on DagsHub).

Every training run is logged with the code version (git commit), the data version
(DVC), its settings, its test results and the trained model. The full model is
registered as a new version of "lake-geneva-temp", and the alias "champion" moves to
it only if it passes the promotion gate (gate.py).

    python src/lakegeneva/training/track.py --promote <version>   # promote by hand
"""
# Based on the MLOps HS26 Block 1 exercise solution (mlops-quotes, train.py), F. Bär
# [5] in README.md: log the parameters, the git commit, the metrics and the model,
# register every model, and move the alias "champion" to the one that is used.
# Changed for this project: the data version from DVC is logged too, the evaluation
# tables are saved with the run, the signature names the model's output (a change,
# not a temperature), and a gate decides about the alias.
#
# Where the runs go: to the address in the environment variable MLFLOW_TRACKING_URI
# (DagsHub: https://dagshub.com/<user>/lake-geneva-temp.mlflow, with the user name and
# a token in MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD). Without it, MLflow
# writes to a local database mlflow.db and the folder mlruns/ (both not in Git), which
# is how the code is tested on a laptop. `uv run mlflow ui` shows the local runs.
#
# MLflow is only in the dev group of pyproject.toml (like in the exercise): the Docker
# image of the feature pipeline does not need it. That is why it is imported inside the
# functions and not at the top of this file.

import argparse  # reads the command-line option --promote

import pandas as pd  # the evaluation tables

from lakegeneva.common import baselines, forecasting  # the frozen trend factor; the name of the model's output (a change)
from lakegeneva.common.feature_store import git, is_proven  # runs git; is the data version proven?
from lakegeneva.training import gate  # the four checks that decide about the alias

# The MLflow experiment (a folder of runs) and the registered model.
EXPERIMENT = "lake-geneva-temp"
MODEL_NAME = "lake-geneva-temp"
# The alias of the model that the live forecast will use (MS4).
CHAMPION = "champion"
# Short metric names for the windows of train.evaluate().
WINDOW_NAMES = {"test, all year": "test", "test, change > 0.8": "big_change", "test, Oct-Jan": "oct_jan"}
COLUMN_NAMES = {
    "days": "days",
    # Only in the big-change window: runs of consecutive big-change days (review point B3).
    "events": "events",
    "model": "mae",
    "persistence": "mae_persistence",
    # The target baseline (frozen factor, param trend_factor). Runs before 20.11.2026
    # logged here the factor 1/3, and their target came from the better baseline.
    "trend_adjusted": "mae_trend_adjusted",
    "target": "target",
    "gain_%": "gain_pct",
    "diff": "diff",
    "ci_low": "ci_low",
    "ci_high": "ci_high",
}


def git_commit() -> str:
    """The commit of the code that trained the model; "-dirty" if a tracked file was changed."""
    commit = git("rev-parse", "HEAD")
    if not commit:
        return "no-git"
    # --untracked-files=no: only changes to files that are in Git count. A run from
    # changed code cannot be repeated from the commit alone, so it gets a mark.
    changed = git("status", "--porcelain", "--untracked-files=no")
    return commit + ("-dirty" if changed else "")


def flat_metrics(evaluation: pd.DataFrame) -> dict:
    """The evaluation table as single numbers for MLflow, e.g. test_mae or big_change_ci_high."""
    metrics = {}
    for window, prefix in WINDOW_NAMES.items():
        if window not in evaluation.index:
            continue
        for column, name in COLUMN_NAMES.items():
            value = evaluation.loc[window].get(column)
            if value is not None and pd.notna(value):  # windows without an interval are skipped
                metrics[f"{prefix}_{name}"] = float(value)
    return metrics


def log_run(name: str, result: dict, rows: pd.DataFrame, versions: dict, summary: dict) -> str:
    """Log one trained model with everything needed to repeat it; return its model URI."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature  # builds the signature from example rows

    columns = result["columns"]
    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "feature_set": name,
                "features": ",".join(columns),
                "alpha": result["alpha"],
                # What the model predicts; the forecast is the last reading plus this
                # (forecasting.temperature(), the same step in the test and live).
                "target": "tomorrow's mean minus the last reading before 18:00",
                # The baseline of the 20 % target, frozen on the validation days (MS2
                # review point B1). With it, runs from before and after the change can
                # be told apart: the older runs have no trend_factor.
                "target_baseline": "water_last + trend_factor * water_trend_3d",
                "trend_factor": baselines.TREND_FACTOR,
                "training_rows": len(rows),
                "first_row": rows.index.min().date().isoformat(),
                "last_row": rows.index.max().date().isoformat(),
                **versions,
                # The validation days of the gate (gate.validation_summary) and their
                # fingerprint: the next run can compare its validation MAE with this
                # one only if the fingerprint is the same (check G2).
                **{key: summary[key] for key in ("validation_days", "validation_first", "validation_last", "validation_md5")},
            }
        )
        # Test results, and the validation error of every alpha that was tried.
        mlflow.log_metrics(flat_metrics(result["evaluation"]))
        mlflow.log_metrics({f"validation_mae_alpha_{alpha:g}": error for alpha, error in result["validation_mae"].items()})
        # The validation MAE of the procedure and of the two free forecasts (check G1).
        mlflow.log_metrics({key: summary[key] for key in ("validation_mae", "validation_mae_persistence", "validation_mae_row3")})
        # The tables behind the numbers, to read or compare later.
        mlflow.log_text(result["evaluation"].to_csv(), "evaluation.csv")
        mlflow.log_text(result["forecast"].to_frame().join(rows[["water_last", "label_water_mean_d1"]]).to_csv(), "test_forecasts.csv")
        # The final model, trained on all rows. input_example: a few real rows, so the
        # model page shows which columns the model expects (its "signature").
        example = rows[columns].head(3)
        # The signature also names the OUTPUT. The registered model returns tomorrow's
        # change (about -0.3 degC), not the temperature. Without a name, MLflow only
        # says "a number", and a reader could take it for the temperature (found in a
        # review on 15.11.2026). infer_signature reads the column names of the example
        # rows and the name of the example answers; the name is "change_degC". It
        # changes nothing in the model or its numbers, only the text on the model page.
        # The temperature is made by forecasting.temperature_for_rows(model, rows).
        answers = pd.Series(result["model"].predict(example), name=forecasting.MODEL_OUTPUT)
        signature = infer_signature(example, answers)
        info = mlflow.sklearn.log_model(result["model"], name="model", input_example=example, signature=signature)
        return info.model_uri


def meets_target(evaluation: pd.DataFrame) -> bool:
    """The success criterion of the README: 20 % below the target baseline on all test days AND on the big-change days."""
    # The README has two targets, and both must be met (found in a review on 14.11.2026:
    # the first version checked only all test days). A window that is missing or has
    # no MAE counts as not met: a target cannot be met on days that were not tested.
    for window in ("test, all year", "test, change > 0.8"):
        if window not in evaluation.index:
            return False
        row = evaluation.loc[window]
        model, target = row.get("model"), row.get("target")
        if model is None or target is None or pd.isna(model) or pd.isna(target) or model > target:
            return False
    return True


def register(model_uri: str) -> str:
    """Register the model as a new version (without moving the alias); return the version."""
    import mlflow

    # str(): MLflow 3 gives the version as a number; the registry calls take text too.
    return str(mlflow.register_model(model_uri, MODEL_NAME).version)


def champion_version() -> str | None:
    """The version that has the alias "champion" right now, e.g. "3". None if there is none."""
    import mlflow
    from mlflow.exceptions import MlflowException

    client = mlflow.MlflowClient()
    # "No champion" must be CONFIRMED: only then may checks G2 and G3 be skipped. Any
    # other problem (DagsHub does not answer) is raised, and the gate says "error"
    # instead of "passed" (fail-closed, review of 17.11.2026).
    try:
        aliases = client.get_registered_model(MODEL_NAME).aliases  # e.g. {"champion": "3"}
    except MlflowException as error:
        # The only confirmed case: the registered model does not exist at all yet.
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise
    # .get(): None if the model exists but no version has the alias yet.
    return aliases.get(CHAMPION)


def champion_info() -> dict | None:
    """The current champion: its model, and the parameters and metrics of its run. None if there is none."""
    import mlflow
    import mlflow.sklearn  # loads the scikit-learn model

    number = champion_version()
    if number is None:
        return None
    client = mlflow.MlflowClient()
    version = client.get_model_version(MODEL_NAME, number)
    run = client.get_run(version.run_id)  # the run that trained the champion
    return {
        "version": version.version,
        # Loaded by its version NUMBER, not by the alias: if a person moved the alias in
        # between (track.py --promote), "@champion" could load another version than the
        # one whose metrics were just read, and G2 and G3 would compare two different
        # models (found in a review on 17.11.2026).
        "model": mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version.version}"),
        "params": run.data.params,
        "metrics": run.data.metrics,
    }


def decide(version: str, checks: dict, error: str | None, target_met: bool, checked_against: str | None) -> bool:
    """Write the result of the gate as tags of the version; move the alias if it passed.

    Returns (promoted?, error): the error can be new here (the champion changed).

    checked_against: the champion version the checks compared with (None: there was none).
    """
    import mlflow

    client = mlflow.MlflowClient()
    promote = error is None and gate.passes(checks)
    if promote:
        # The alias is read again right before it is moved. If a person moved it while
        # the gate ran (for example a rollback with track.py --promote), the checks were
        # made against another champion: this version is not promoted, and the next run
        # compares with the new champion. Without this, a retrain could undo a rollback
        # (review of 18.11.2026). str(): MLflow gives versions as numbers or text.
        try:
            now = champion_version()
        except Exception as problem:  # cannot read the alias: fail-closed
            now, error = "unknown", f"{type(problem).__name__}: {problem}"
        if str(now) != str(checked_against):
            promote = False
            error = error or f"the champion changed while the gate ran (checked against v{checked_against}, now v{now})"
    # "error": the gate could not check (not the same as "failed": the gate checked and said no).
    client.set_model_version_tag(MODEL_NAME, version, "gate", "passed" if promote else ("error" if error else "failed"))
    # One tag per check, so the model page shows WHY: e.g. "failed: largest difference 1.2 degC".
    for name, text in checks.items():
        client.set_model_version_tag(MODEL_NAME, version, f"gate_{name}", text)
    if error:
        client.set_model_version_tag(MODEL_NAME, version, "gate_error", error[:4000])
    # The gate is not the success criterion: a champion passed the gate, but it may still
    # miss the 20 % target. This tag says which.
    client.set_model_version_tag(MODEL_NAME, version, "target_20_percent", "met" if target_met else "not met")
    if promote:
        # Moving the alias is the promotion: the live forecast loads models:/<name>@champion.
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    return promote, error


def track(results: dict, rows: pd.DataFrame, data: dict, table: pd.DataFrame) -> dict:
    """Log both models of train.run(); register the full model and promote it if it passes the gate.

    rows: the rows with all features and a label (train.model_rows).
    data: the data version of the table, from feature_store.load_features().
    table: the whole feature table, for the checks on the newest days and the issue day.
    Returns the version, whether it was promoted, and the result of each check.
    """
    import mlflow

    mlflow.set_experiment(EXPERIMENT)
    # data: the version of the table that train.py really read (feature_store.py), with
    # the hash in data/features.dvc and the data tag (review point E1).
    versions = {"git_commit": git_commit(), **data}
    decision = {}
    for name, result in results.items():
        # The validation days of the gate, for both models (logged for comparison).
        summary = gate.validation_summary(rows, result["columns"], result["alpha"])
        model_uri = log_run(name, result, rows, versions, summary)
        # Only the full model is a candidate for the live forecast; the water-only model
        # is logged for the comparison (review point B2).
        if name != "full":
            continue
        # Registered only on a proven data version: the DVC list confirms the table and
        # a commit or tag gives it back. Otherwise the model would point to no data
        # version (review point E1; registered with "unknown" in a review on 24.11.2026).
        # The runs above stay logged, so the numbers can still be compared.
        if not is_proven(versions):
            reason = (
                f"data version not proven (DVC match: {versions['data_matches_dvc']}, "
                f"commit: {versions['data_commit']}, tag: {versions['data_tag']}). For the checked-out table: "
                f"commit the pointer and run dvc pull. For an older version (--data-version): its DVC file list "
                f"is not on this computer, so it can be trained on and logged, but not registered"
            )
            decision = {"version": None, "promoted": False, "checks": {}, "error": reason}
            print(f"NOT registered: {reason}")
            continue
        version = register(model_uri)
        checks, error, checked_against = {}, None, None
        try:
            # The checks run on the model as it was loaded BACK from the registry: the
            # artifact that would become the champion (review of 17.11.2026).
            candidate = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
            champion = champion_info()
            checked_against = champion["version"] if champion else None
            # rows.index.max(): the newest row with a label, for the freshness check G5.
            checks = gate.run_checks(summary, candidate, champion, table, rows.index.max())
        except Exception as problem:
            # On purpose ANY problem: fail-closed, a check that could not run must never
            # count as passed.
            error = f"{type(problem).__name__}: {problem}"
        target_met = meets_target(result["evaluation"])
        promoted, error = decide(version, checks, error, target_met, checked_against)
        decision = {"version": version, "promoted": promoted, "checks": checks, "error": error}
        print(f"registered {MODEL_NAME} v{version}; gate {'passed: alias champion -> v' + version if promoted else 'NOT passed: champion unchanged'}; 20 % target {'met' if target_met else 'not met'}")
        for check, text in checks.items():
            print(f"  {check}: {text}")
        if error:
            print(f"  gate error: {error}")
    print(
        f"data: {versions['features_file']} (md5 {versions['features_md5'][:8]}), DVC version: "
        f"{versions['data_matches_dvc']}, tag: {versions['data_tag']}, commit: {versions['data_commit'][:7]}"
    )
    print(f"runs logged to {mlflow.get_tracking_uri()}")
    return decision


def promote_by_hand(version: str) -> None:
    """Move the alias "champion" to a version after a person checked it."""
    # When the gate cannot decide (for example the champion was trained before this gate
    # existed, so its validation MAE has no fingerprint, or old labels changed), the
    # new version is not promoted. A person looks at the tags and the numbers of the
    # version in MLflow and, if it is fine, promotes it with this command. The tag
    # says that a person decided, not the gate.
    # Normally it runs in the workflow promote.yml, which waits while a retrain runs
    # (same concurrency group): started from a laptop during a retrain, the retrain
    # could overwrite it between its last check and its alias write (review of
    # 20.11.2026). So from a laptop only when no Retrain run is in progress.
    import mlflow

    client = mlflow.MlflowClient()
    client.get_model_version(MODEL_NAME, version)  # stops with an error if the version does not exist
    client.set_model_version_tag(MODEL_NAME, version, "gate", "promoted by hand")
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    print(f"alias {CHAMPION} -> {MODEL_NAME} v{version} (promoted by hand)")


def main() -> None:
    # Command-line options; __doc__ (the text at the top) becomes the --help text.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--promote", required=True, metavar="VERSION", help="the model version to promote by hand")
    args = parser.parse_args()
    promote_by_hand(args.promote)


# Runs main() only when the file is started as a script, not when it is imported
# (train.py and the tests import this file).
if __name__ == "__main__":
    main()
