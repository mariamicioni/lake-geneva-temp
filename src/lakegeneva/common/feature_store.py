"""The feature store: read the feature table of one data version, and say which version it is.

Training reads the table only through load_features(), and so will the live forecast
in MS4. Every MLflow run logs the version that load_features() names, and a model is
registered only if that version is proven (track.py), so each registered model
points to exactly one data version.

    table, data = load_features()              # the table that is checked out (after dvc pull)
    table, data = load_features("data-v1")     # an older version: a tag or a commit
"""
# Why one read function (MS2 review point E1, also A1): before, train.py, experiments.py
# and check_fresh.py each opened the Parquet file themselves, and nothing could load an
# older data version by name. Now there is one place that reads the table and one
# place that decides which version it is.
# How a version is found: Git keeps the DVC pointer data/features.dvc of every version
# (one commit per evening), DVC keeps the table itself in the bucket. A version is
# therefore a Git name (a tag like data-v1 or a commit), and `dvc get --rev` fetches
# the table that the pointer of that commit names. That needs the bucket key, like
# `dvc pull`. Without a version, the table that is already checked out is read.

import hashlib  # md5 of the table that was really read
import io  # reads the table from the bytes that were hashed
import json  # reads the DVC list of files of a data version
import re  # the exact form of a DVC pointer
import shutil  # finds out whether git is installed
import subprocess  # runs git and dvc
import tempfile  # a folder for a table fetched with dvc get, deleted afterwards
from pathlib import Path  # file paths that work on Windows and Linux

import pandas as pd  # reads the Parquet table

# --- Settings -----------------------------------------------------------------
# The table of the checked-out version (written by build_features.py).
FEATURES_FILE = Path("data/features/features.parquet")
# The DVC pointer of the table: its md5 is the version of the training data.
FEATURES_POINTER = Path("data/features.dvc")
# Where DVC keeps its list of the files of each data version (default cache of DVC 3).
DVC_CACHE = Path(".dvc/cache/files/md5")
# The names of the data tags, e.g. data-v1 or data-2026-11-24 (review point A1).
TAG_PATTERN = "data-*"
# What git_available() is false for: no git on this computer, or no Git repository here.
NO_GIT = "unknown"
# The only form of data/features.dvc that is accepted (see pointer_md5): the six lines
# DVC 3 writes for one output folder "features", one pattern per line.
POINTER_FORM = (
    re.compile(r"outs:"),
    re.compile(r"- md5: ([0-9a-f]{32}\.dir)"),
    re.compile(r"  size: \d+"),
    re.compile(r"  nfiles: \d+"),
    re.compile(r"  hash: md5"),
    re.compile(r"  path: features"),
)


def git(*args: str, input: str | None = None, strip: bool = True) -> str:
    """The output of one git command, or "" if it fails or git is not available.

    strip=False keeps spaces at the ends (for lists of names, e.g. tags).
    """
    try:
        # encoding="utf-8": Git writes UTF-8; Windows would otherwise read it as CP1252 and
        # a tag name with an accent would not match (review of 24.11.2026).
        output = subprocess.check_output(
            ["git", *args], text=True, encoding="utf-8", errors="replace", input=input, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return output.strip() if strip else output


def git_available() -> bool:
    """Is git installed, and is the current folder inside a Git repository?"""
    # Without this check a missing git would look like "no tag" or "unknown version"
    # (review of 24.11.2026): the answers must say that they could not be checked.
    return shutil.which("git") is not None and git("rev-parse", "--is-inside-work-tree") == "true"


def file_md5(path: Path) -> str:
    """The md5 of a file's bytes (the same number DVC uses for a file)."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def pointer_md5(text: str) -> str:
    """The data version named in a DVC pointer: its "<hash>.dir" ("" if the pointer has another form)."""
    # Only a pointer in exactly the form DVC writes for the feature table counts: the
    # six lines below, in this order, nothing else. Every other form is rejected, even
    # valid YAML: a renamed output ("path: archive"), a moved working folder ("wdir:"),
    # the same key in quotes ('"wdir": archive') or any other field could make DVC fetch
    # the table from somewhere else, so a table left in data/features/ would look like
    # this version (reviews of 24.11.2026). All 31 pointers in the history up to then
    # have exactly this form. If a future DVC version writes another form, training is
    # still logged, but no model is registered and the run turns red: then this list
    # has to be checked again.
    lines = [line for line in text.replace("\r\n", "\n").split("\n") if line != ""]
    if len(lines) != len(POINTER_FORM) or not all(form.fullmatch(line) for form, line in zip(POINTER_FORM, lines)):
        return ""
    return POINTER_FORM[1].fullmatch(lines[1]).group(1)


def tag_pointers() -> dict:
    """Every data tag and the hash its pointer names, with two git commands for all tags."""
    # One git process per tag took 75 ms on Windows, about 27 s for a year of daily tags
    # (review of 24.11.2026). `git cat-file --batch` reads all pointers in one process:
    # for each line "<tag>:data/features.dvc" it answers "<id> blob <size>" and the
    # file, or "<name> missing" if that tag has no pointer.
    # Git ends each name with a line feed ("\n") and nothing else. So the list is cut
    # only there: not at other spaces or Unicode line separators that a tag name may
    # contain, and spaces at the end of a name are kept (review of 24.11.2026).
    tags = [tag for tag in git("tag", "--list", TAG_PATTERN, strip=False).split("\n") if tag]
    if not tags:
        return {}
    # refs/tags/<tag>^{commit}: the commit the tag stands for. A tag that points to
    # something else (e.g. directly to a Git tree) gives "missing": load_features()
    # could not use it, so it must never count as a way back to a version (review of
    # 24.11.2026). refs/tags/ also keeps a branch with the same name out.
    request = "".join(f"refs/tags/{tag}^{{commit}}:{FEATURES_POINTER.as_posix()}\n" for tag in tags)
    answer = subprocess.run(
        ["git", "cat-file", "--batch"], input=request.encode("utf-8"), capture_output=True, check=True
    ).stdout
    pointers, position = {}, 0
    for tag in tags:
        end = answer.index(b"\n", position)
        header = answer[position:end].decode("utf-8").split()
        position = end + 1
        if header[-1] == "missing":
            pointers[tag] = ""
            continue
        size = int(header[2])
        pointers[tag] = pointer_md5(answer[position : position + size].decode("utf-8"))
        position += size + 1  # the file and the newline after it
    return pointers


def data_tag(dir_md5: str) -> str:
    """The data tags (data-...) that name this data version, "none", or "unknown" without git."""
    # A tag names a data version if the DVC pointer at the tagged commit has the same
    # hash. Not "the tag on the commit that stored the pointer": data-v1 sits on a later
    # commit than the one that stored its pointer (found on 24.11.2026). Several tags
    # of the same version are joined with commas.
    if not git_available():
        return NO_GIT
    tags = [tag for tag, md5 in tag_pointers().items() if md5 == dir_md5]
    return ",".join(sorted(tags)) or "none"


def matches_dvc(md5: str, dir_md5: str, cache: Path = DVC_CACHE) -> str:
    """Is a table with this md5 the table of the data version dir_md5? "yes", "no" or "unknown"."""
    # The pointer names a list of files ("<hash>.dir"); DVC keeps that list in its
    # cache, and the table must be in the list with the same md5. Without the list
    # (never pulled on this computer) the answer is "unknown", never a guess.
    listing = cache / dir_md5[:2] / dir_md5[2:] if dir_md5 else None
    if listing is None or not listing.exists():
        return "unknown"
    # DVC stores each list under its own md5 (checked on 24.11.2026 for every list in
    # the cache). A list whose bytes have another md5 is not this version's list (e.g.
    # copied under the wrong name), so it proves nothing (review of 24.11.2026).
    content = listing.read_bytes()
    if hashlib.md5(content).hexdigest() + ".dir" != dir_md5:
        return "unknown"
    expected = {entry["relpath"]: entry["md5"] for entry in json.loads(content.decode("utf-8"))}
    return "yes" if expected.get(FEATURES_FILE.name) == md5 else "no"


def read_once(path: Path) -> tuple[pd.DataFrame, str]:
    """The table and the md5 of exactly the bytes it was read from."""
    # Read the file once and use the same bytes for both: reading it twice (once for
    # pandas, once for the md5) could give the md5 of another table if the file is
    # replaced in between, e.g. by a dvc pull (review of 24.11.2026).
    content = Path(path).read_bytes()
    return pd.read_parquet(io.BytesIO(content)), hashlib.md5(content).hexdigest()


def data_version(features: Path, md5: str | None = None, pointer: Path = FEATURES_POINTER, cache: Path = DVC_CACHE) -> dict:
    """Which data version a checked-out table is: proven, or marked as unknown or not DVC.

    md5: the md5 of the table that was read (from read_once); without it the file is hashed.
    """
    features = Path(features)
    md5 = md5 or file_md5(features)
    # Always true: the name and the md5 of the table that was read.
    version = {"features_file": features.as_posix(), "features_md5": md5}
    # The DVC version is claimed only if it is proven (see matches_dvc): otherwise a run
    # with another table (--features) or a changed local table would name the wrong
    # data version (found in a review on 14.11.2026).
    dir_md5 = pointer_md5(pointer.read_text(encoding="utf-8")) if pointer.exists() else ""
    match = matches_dvc(md5, dir_md5, cache)
    if match == "unknown":
        return version | {"data_md5": "unknown", "data_commit": "unknown", "data_tag": "unknown", "data_matches_dvc": "unknown"}
    if match == "no":
        return version | {"data_md5": "not a DVC version", "data_commit": "none", "data_tag": "none", "data_matches_dvc": "no"}
    # `git checkout <data_commit>` and then `dvc checkout` give back exactly this table,
    # but only if that commit stores exactly the hash that was just verified. The
    # newest commit of the pointer is not enough: the pointer can be switched between
    # the two steps, or Git can be told to ignore changes to it (assume-unchanged), and
    # then the run would name the commit of another table (review of 24.11.2026).
    # In a shallow clone (git clone --depth 1) Git cannot see older commits, so the
    # commit named can be a later one that stores the same pointer: it gives back the
    # same table, but it is not the evening that made it. Both workflows check out the
    # full history (fetch-depth: 0).
    if not git_available():
        data_commit = NO_GIT
    else:
        data_commit = git("log", "-1", "--format=%H", "--", pointer.as_posix())
        if not data_commit or pointer_md5(git("show", f"{data_commit}:{pointer.as_posix()}")) != dir_md5:
            # Not (yet) a commit anyone can check out. Its tags still count: a tag with
            # the same hash gives back exactly this table.
            data_commit = "not committed"
    return version | {"data_md5": dir_md5, "data_commit": data_commit, "data_tag": data_tag(dir_md5), "data_matches_dvc": "yes"}


def fetch(commit: str, out: Path) -> None:
    """Save the table of the data version stored at `commit` as `out` (dvc get, needs the key)."""
    # "." is this repository; --rev reads the pointer of that commit and downloads the
    # table it names from the bucket. The checked-out data/ folder is not touched.
    command = ["dvc", "get", ".", FEATURES_FILE.as_posix(), "--rev", commit, "-o", str(out)]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as problem:
        raise RuntimeError("dvc is not installed: uv tool install \"dvc[gs]==3.67.1\" (see README)") from problem


def load_features(version: str | None = None, path: Path | str = FEATURES_FILE) -> tuple[pd.DataFrame, dict]:
    """The feature table of one data version, and the facts that name that version.

    version None: the table at `path` (the checked-out version, after `dvc pull`).
    version "data-v1", another tag or a commit: the table of that version (dvc get).
    The second value is what MLflow logs: data_md5 (the hash in data/features.dvc),
    data_commit, data_tag, features_md5 and whether the table matches DVC.
    """
    if version is None:
        path = Path(path)
        table, md5 = read_once(path)
        return table, data_version(path, md5) | {"data_requested": "checked out"}

    # A version is a Git name, so without git it cannot be found (not "unknown version").
    if not git_available():
        raise RuntimeError("git is not available here, so the data version cannot be looked up.")
    # The commit that the name stands for (^{commit}: also for an annotated tag such as
    # data-v1); an unknown name stops here with a clear text.
    commit = git("rev-parse", "--verify", "--quiet", f"{version}^{{commit}}")
    if not commit:
        raise ValueError(f"Unknown data version {version!r}: give a tag such as data-v1 or a commit.")
    # The version is the pointer as it was at that commit, stored by the newest commit
    # up to there that changed the pointer.
    dir_md5 = pointer_md5(git("show", f"{commit}:{FEATURES_POINTER.as_posix()}"))
    if not dir_md5:
        raise ValueError(f"Data version {version!r} has no feature table (no {FEATURES_POINTER} at {commit[:7]}).")
    data_commit = git("log", "-1", "--format=%H", commit, "--", FEATURES_POINTER.as_posix())
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / FEATURES_FILE.name
        fetch(commit, out)
        table, md5 = read_once(out)
    # dvc get read the table that this pointer names; the check against the DVC list is
    # still made. A table that is NOT in the list is a wrong download: it must never be
    # trained on under this version's name (review of 24.11.2026).
    match = matches_dvc(md5, dir_md5)
    if match == "no":
        raise ValueError(f"The table fetched for {version!r} is not the one in its DVC list (md5 {md5}).")
    return table, {
        "features_file": f"{FEATURES_FILE.as_posix()} at {version}",
        "features_md5": md5,
        "data_md5": dir_md5,
        "data_commit": data_commit,
        "data_tag": data_tag(dir_md5),
        "data_matches_dvc": match,
        "data_requested": version,
    }


def is_proven(data: dict) -> bool:
    """Can this data version be named exactly: the DVC list confirms it, and a commit or tag gives it back?"""
    # A model may be registered only on such a version (track.py, review of 24.11.2026).
    # "not committed" counts if a tag with the same hash exists: the tag gives it back.
    recoverable = len(data.get("data_commit", "")) == 40 or data.get("data_tag") not in ("none", "unknown", None)
    return data.get("data_matches_dvc") == "yes" and recoverable
