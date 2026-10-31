"""One rule for all download scripts: a raw file is never overwritten."""
# Why: the file names hold the download day or the date range, so a second run on the
# same UTC day (a manual re-run, or a scheduled run that GitHub starts very late) wants
# the same name. If its answer had gaps, the good values of the first file would be gone
# before features.py could keep them. So every download gets its own file, and the
# loaders in features.py decide which value counts. The newest forecast (audit copy)
# can never be downloaded again, so there an overwrite would lose it for good.

from pathlib import Path  # file paths that work on Windows and Linux


def new_path(path: Path) -> Path:
    """Return path if that name is still free, otherwise the same name with _run02, _run03 ... added."""
    candidate = path
    number = 1
    while candidate.exists():
        number += 1
        # stem = the name without ".csv". Two digits keep the name order right after run
        # 9: load_water() reads the files in name order, and the last one wins, so
        # "..._2026-10-31.csv" must come before "..._run02.csv", and "..._run09.csv"
        # before "..._run10.csv" (without the 0, "run10" would come before "run9").
        candidate = path.with_name(f"{path.stem}_run{number:02d}{path.suffix}")
    return candidate
