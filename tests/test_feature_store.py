"""Tests for the feature store: one read path, and the data version it names (MS2 review point E1)."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
# The tests that need Git make a small repository in a temporary folder; nothing is
# downloaded (dvc get is replaced by a fake that writes a table). The fake DVC file
# lists are stored like DVC stores them: under the md5 of their own bytes.
import hashlib
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from lakegeneva.common import feature_store

# The table of the first stored version in the test repository.
TABLE_A = pd.DataFrame({"water_last": [12.0]}, index=pd.to_datetime(["2026-11-23"]))


def parquet_bytes(table: pd.DataFrame) -> bytes:
    """The table as Parquet bytes (the same bytes every time in one test run)."""
    buffer = io.BytesIO()
    table.to_parquet(buffer)
    return buffer.getvalue()


def dvc_list(table_md5: str) -> tuple[str, bytes]:
    """DVC's file list of a version with this table, and its name: the md5 of its bytes + ".dir"."""
    content = json.dumps([{"md5": table_md5, "relpath": "features.parquet"}]).encode("utf-8")
    return hashlib.md5(content).hexdigest() + ".dir", content


def pointer_text(dir_md5: str, path: str = "features") -> str:
    """A DVC pointer exactly like data/features.dvc (the six lines DVC 3 writes)."""
    return f"outs:\n- md5: {dir_md5}\n  size: 257841\n  nfiles: 1\n  hash: md5\n  path: {path}\n"


# The hash of a second data version (any 32 hex digits + ".dir").
OTHER = "f" * 32 + ".dir"


def put_list(cache, dir_md5: str, content: bytes) -> None:
    """Store a file list in a DVC cache folder under the given name."""
    (cache / dir_md5[:2]).mkdir(parents=True, exist_ok=True)
    (cache / dir_md5[:2] / dir_md5[2:]).write_bytes(content)


def write_table(path, content: bytes) -> None:
    """What the fake dvc get does: write the bytes of a table to the output file."""
    path.write_bytes(content)


def fake_dvc(folder, content: bytes):
    """A feature file, a DVC pointer and the DVC list of files of that version (no Git)."""
    features = folder / "features.parquet"
    features.write_bytes(content)
    dir_md5, listing = dvc_list(feature_store.file_md5(features))
    pointer = folder / "features.dvc"
    pointer.write_text(pointer_text(dir_md5), encoding="utf-8")
    cache = folder / "cache"
    put_list(cache, dir_md5, listing)
    return features, pointer, cache, dir_md5


def run_git(*args: str) -> str:
    """One git command in the current folder, with a fixed author (no user settings needed)."""
    command = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args]
    return subprocess.check_output(command, text=True, encoding="utf-8").strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A tiny Git repository with one stored data version (table A) and the annotated tag data-2026-11-24.

    The DVC list of the version is NOT in the cache yet: put_list(repo.cache, ...) adds it.
    """
    monkeypatch.chdir(tmp_path)
    run_git("init", "-q")
    # A name and email in the repository itself, like the daily job sets them: Git needs
    # them for commits and annotated tags, and CI has no global ones (found on 25.11.2026:
    # the tag tests passed on the laptop with its own Git name, but failed on GitHub).
    run_git("config", "user.name", "test")
    run_git("config", "user.email", "test@example.com")
    table_bytes = parquet_bytes(TABLE_A)
    dir_md5, listing = dvc_list(hashlib.md5(table_bytes).hexdigest())
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "features.dvc").write_text(pointer_text(dir_md5), encoding="utf-8")
    run_git("add", "data/features.dvc")
    run_git("commit", "-q", "-m", "chore(data): daily update 2026-11-24")
    # Annotated, like data-v1: the tag is its own Git object, not the commit itself.
    run_git("tag", "-a", "data-2026-11-24", "-m", "data version of 24.11.2026")
    return SimpleNamespace(
        path=tmp_path,
        dir=dir_md5,  # the hash in data/features.dvc
        listing=listing,  # DVC's file list of the version
        table_bytes=table_bytes,  # the bytes of table A
        cache=tmp_path / ".dvc" / "cache" / "files" / "md5",
        pointer=tmp_path / "data" / "features.dvc",
    )


def check_out_table_a(repo) -> None:
    """Table A in data/features/ and its DVC list in the cache, as after `dvc pull`."""
    path = repo.path / "data" / "features" / "features.parquet"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(repo.table_bytes)
    put_list(repo.cache, repo.dir, repo.listing)


# --- The version of a checked-out table -----------------------------------------------


def test_data_version_is_claimed_when_the_file_is_the_dvc_version(tmp_path):
    features, pointer, cache, dir_md5 = fake_dvc(tmp_path, b"the stored table")

    version = feature_store.data_version(features, pointer=pointer, cache=cache)

    assert version["data_md5"] == dir_md5
    assert version["data_matches_dvc"] == "yes"
    assert version["features_md5"] == feature_store.file_md5(features)


def test_data_version_is_not_claimed_for_another_or_changed_table(tmp_path):
    features, pointer, cache, _ = fake_dvc(tmp_path, b"the stored table")
    # The table was changed after DVC stored it (or another table was given).
    features.write_bytes(b"a changed table")

    version = feature_store.data_version(features, pointer=pointer, cache=cache)

    assert version["data_matches_dvc"] == "no"
    assert version["data_md5"] == "not a DVC version"
    assert version["data_tag"] == "none"
    # The md5 of the file that was really used is still logged.
    assert version["features_md5"] == feature_store.file_md5(features)


def test_data_version_is_unknown_without_the_dvc_list(tmp_path):
    features, pointer, _, _ = fake_dvc(tmp_path, b"the stored table")

    version = feature_store.data_version(features, pointer=pointer, cache=tmp_path / "empty_cache")

    assert version["data_matches_dvc"] == "unknown"
    assert version["data_md5"] == "unknown"


def test_a_list_stored_under_the_wrong_name_proves_nothing(tmp_path):
    features, pointer, cache, dir_md5 = fake_dvc(tmp_path, b"the stored table")
    # Another version's list, copied under this version's name. It even lists the table
    # that was read, but its own md5 is not its name, so it is not this version's list.
    other = json.dumps([{"md5": feature_store.file_md5(features), "relpath": "features.parquet", "x": 1}]).encode()
    put_list(cache, dir_md5, other)

    version = feature_store.data_version(features, pointer=pointer, cache=cache)

    assert version["data_matches_dvc"] == "unknown"
    assert not feature_store.is_proven(version)


def test_a_pointer_with_another_output_is_not_the_feature_table(tmp_path):
    features, pointer, cache, dir_md5 = fake_dvc(tmp_path, b"the stored table")
    # The pointer's output was renamed: DVC would fetch the table only as archive/, so a
    # table left in data/features/ is not this version.
    pointer.write_text(pointer_text(dir_md5, path="archive"), encoding="utf-8")

    version = feature_store.data_version(features, pointer=pointer, cache=cache)

    assert version["data_matches_dvc"] == "unknown"
    assert feature_store.pointer_md5(pointer_text(dir_md5, path="archive")) == ""


def test_a_checked_out_table_names_its_commit_and_tag(repo):
    check_out_table_a(repo)

    read, data = feature_store.load_features()

    pd.testing.assert_frame_equal(read, TABLE_A)
    assert data["data_requested"] == "checked out"
    assert data["data_md5"] == repo.dir  # the hash in data/features.dvc
    assert data["data_commit"] == run_git("rev-parse", "HEAD")
    assert data["data_tag"] == "data-2026-11-24"
    assert data["data_matches_dvc"] == "yes"
    assert feature_store.is_proven(data)


def test_the_hash_belongs_to_exactly_the_table_that_was_read(tmp_path, monkeypatch):
    first = pd.DataFrame({"water_last": [12.0]})
    second = pd.DataFrame({"water_last": [99.0]})
    path = tmp_path / "features.parquet"
    first.to_parquet(path)
    first_md5 = feature_store.file_md5(path)
    read_parquet = pd.read_parquet

    def read_and_replace(source, *args, **kwargs):
        # Someone replaces the file right after it was read (e.g. a dvc pull).
        table = read_parquet(source, *args, **kwargs)
        second.to_parquet(path)
        return table

    monkeypatch.setattr(feature_store.pd, "read_parquet", read_and_replace)

    table, data = feature_store.load_features(path=path)

    # The md5 is the one of the table that was returned, not of the new file.
    pd.testing.assert_frame_equal(table, first)
    assert data["features_md5"] == first_md5


def test_an_uncommitted_pointer_keeps_its_tag(repo):
    # A second version is committed, then the pointer is set back to the first version
    # without a commit (e.g. after `git checkout data-2026-11-24 -- data/features.dvc`).
    repo.pointer.write_text(pointer_text(OTHER), encoding="utf-8")
    run_git("commit", "-q", "-am", "chore(data): daily update 2026-11-25")
    repo.pointer.write_text(pointer_text(repo.dir), encoding="utf-8")
    check_out_table_a(repo)

    _, data = feature_store.load_features()

    assert data["data_commit"] == "not committed"
    # The tag still gives back exactly this table, so it is named ...
    assert data["data_tag"] == "data-2026-11-24"
    # ... and the version counts as proven.
    assert feature_store.is_proven(data)


def test_the_commit_named_must_store_exactly_the_verified_table(repo):
    # Version A (tagged) is committed, then version B. The checked-out table and pointer
    # are A again, but Git is told to ignore the change (assume-unchanged), so
    # `git status` looks clean and the newest commit of the pointer is B's.
    repo.pointer.write_text(pointer_text(OTHER), encoding="utf-8")
    run_git("commit", "-q", "-am", "chore(data): daily update 2026-11-25")
    commit_b = run_git("rev-parse", "HEAD")
    run_git("update-index", "--assume-unchanged", "data/features.dvc")
    repo.pointer.write_text(pointer_text(repo.dir), encoding="utf-8")
    check_out_table_a(repo)

    _, data = feature_store.load_features()

    # B's commit would give back B's table, not the one that was read: never name it.
    assert data["data_commit"] != commit_b
    assert data["data_commit"] == "not committed"
    # The tag of A is checked on its own and still names exactly this table.
    assert data["data_tag"] == "data-2026-11-24"


# --- Data tags ----------------------------------------------------------------------


def test_the_data_tag_names_the_version_of_its_pointer(repo):
    assert feature_store.data_tag(repo.dir) == "data-2026-11-24"
    # Other tags (e.g. a milestone tag) are not data tags.
    run_git("tag", "ms3")
    assert feature_store.data_tag(repo.dir) == "data-2026-11-24"
    # Another data version has no tag yet.
    assert feature_store.data_tag(OTHER) == "none"


def test_a_tag_on_a_later_commit_with_the_same_pointer_counts(repo, monkeypatch):
    # Like data-v1: the pointer was stored by one commit, the tag sits on a later code
    # commit that did not change the pointer. The tag still names this data version.
    (repo.path / "README.md").write_text("code change", encoding="utf-8")
    run_git("add", "README.md")
    run_git("commit", "-q", "-m", "docs: a code commit")
    run_git("tag", "data-v1")
    monkeypatch.setattr(feature_store, "fetch", lambda commit, out: write_table(out, repo.table_bytes))

    _, data = feature_store.load_features("data-v1")

    assert data["data_tag"] == "data-2026-11-24,data-v1"
    assert data["data_commit"] != run_git("rev-parse", "data-v1")  # the commit that stored the pointer


def test_a_tag_without_a_pointer_is_skipped(repo):
    # A data tag on a commit that has no feature pointer (e.g. set by mistake).
    run_git("rm", "-q", "data/features.dvc")
    run_git("commit", "-q", "-m", "remove the pointer")
    run_git("tag", "data-gone")

    assert feature_store.tag_pointers()["data-gone"] == ""
    assert feature_store.data_tag(repo.dir) == "data-2026-11-24"


def test_a_tag_on_a_tree_is_not_a_way_back(repo):
    # A tag that points directly to a Git tree, not to a commit: load_features() could
    # not use it, so it must not name the version (and so not prove it).
    run_git("tag", "data-tree", "HEAD^{tree}")

    assert feature_store.tag_pointers()["data-tree"] == ""
    assert feature_store.data_tag(repo.dir) == "data-2026-11-24"
    with pytest.raises(ValueError):
        feature_store.load_features("data-tree")


def test_a_tag_name_with_an_accent_is_found(repo):
    # Git writes tag names as UTF-8; on Windows they must not be read as CP1252.
    run_git("tag", "data-révision")

    assert "data-révision" in feature_store.data_tag(repo.dir).split(",")


def test_a_tag_name_with_a_no_break_space_stays_one_name(repo):
    # A no-break space is allowed in a tag name; only line breaks separate the names.
    name = "data-gap HEAD"
    run_git("tag", name)

    tags = feature_store.data_tag(repo.dir).split(",")

    assert name in tags
    assert "HEAD" not in tags and "data-gap" not in tags


def test_the_tag_lookup_needs_three_git_calls_for_any_number_of_tags(repo, monkeypatch):
    for day in range(25, 31):  # six more tags on the same commit
        run_git("tag", f"data-2026-11-{day}")
    calls = []
    run = feature_store.subprocess.run

    def count_run(*args, **kwargs):
        calls.append(args[0][:2])
        return run(*args, **kwargs)

    # Every process goes through subprocess.run (check_output calls it too), so counting
    # run counts each Git process exactly once.
    monkeypatch.setattr(feature_store.subprocess, "run", count_run)

    tags = feature_store.data_tag(repo.dir)

    assert len(tags.split(",")) == 7
    # Is this a Git repository, which tags exist, and all their pointers in one batch:
    # three processes, not one per tag (one per tag took about 27 s for a year of tags).
    assert len(calls) == 3


def test_without_git_the_tag_is_unknown_and_a_version_cannot_be_looked_up(repo, monkeypatch):
    monkeypatch.setattr(feature_store.shutil, "which", lambda name: None)

    # "unknown", not "none": without git nobody checked whether a tag exists.
    assert feature_store.data_tag(repo.dir) == "unknown"
    with pytest.raises(RuntimeError, match="git is not available"):
        feature_store.load_features("data-2026-11-24")


# --- An older version, by name ------------------------------------------------------


def test_an_older_version_is_fetched_by_its_tag(repo, monkeypatch):
    calls = []

    def fake_fetch(commit, out):
        # What `dvc get . data/features/features.parquet --rev <commit> -o <out>` does.
        calls.append(commit)
        out.write_bytes(repo.table_bytes)

    monkeypatch.setattr(feature_store, "fetch", fake_fetch)

    read, data = feature_store.load_features("data-2026-11-24")

    pd.testing.assert_frame_equal(read, TABLE_A)
    # The (annotated) tag is turned into its commit, not the tag object, and that
    # commit's pointer names the version.
    assert calls == [run_git("rev-parse", "HEAD")]
    assert data["data_requested"] == "data-2026-11-24"
    assert data["data_md5"] == repo.dir
    assert data["data_tag"] == "data-2026-11-24"
    # This computer has no DVC list of the version: matching is "unknown", never a guess.
    assert data["data_matches_dvc"] == "unknown"
    # With the list in the cache, the fetched table is proven.
    put_list(repo.cache, repo.dir, repo.listing)
    _, data = feature_store.load_features("data-2026-11-24")
    assert data["data_matches_dvc"] == "yes" and feature_store.is_proven(data)


def test_an_older_version_names_the_commit_that_stored_its_pointer(repo, monkeypatch):
    stored = run_git("rev-parse", "HEAD")
    # A later code commit that does not change the pointer: the data version stays the
    # one stored before, and so does its tag.
    (repo.path / "README.md").write_text("code change", encoding="utf-8")
    run_git("add", "README.md")
    run_git("commit", "-q", "-m", "docs: a code commit")
    monkeypatch.setattr(feature_store, "fetch", lambda commit, out: write_table(out, repo.table_bytes))

    _, data = feature_store.load_features("HEAD")

    assert data["data_commit"] == stored
    assert data["data_tag"] == "data-2026-11-24"


def test_a_wrong_download_stops_instead_of_training_on_it(repo, monkeypatch):
    # The DVC list says the version is table A; dvc get delivers another table.
    put_list(repo.cache, repo.dir, repo.listing)
    monkeypatch.setattr(feature_store, "fetch", lambda commit, out: pd.DataFrame({"a": [1]}).to_parquet(out))

    with pytest.raises(ValueError, match="not the one in its DVC list"):
        feature_store.load_features("data-2026-11-24")


def test_an_unknown_version_stops_with_a_clear_message(repo):
    with pytest.raises(ValueError, match="Unknown data version"):
        feature_store.load_features("data-1999-01-01")


def test_dvc_get_asks_for_the_table_of_that_commit(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(feature_store.subprocess, "run", lambda command, check: seen.append(command))

    feature_store.fetch("abc1234", tmp_path / "out.parquet")

    # Exactly the documented command: this repository, the table, the commit, the output.
    assert seen == [["dvc", "get", ".", "data/features/features.parquet", "--rev", "abc1234", "-o", str(tmp_path / "out.parquet")]]


def test_a_version_is_proven_only_if_dvc_confirms_it_and_it_can_be_got_back():
    proven = {"data_matches_dvc": "yes", "data_commit": "a" * 40, "data_tag": "none"}
    assert feature_store.is_proven(proven)
    # Not confirmed by the DVC list, or nothing that gives it back: not proven.
    assert not feature_store.is_proven(proven | {"data_matches_dvc": "unknown"})
    assert not feature_store.is_proven(proven | {"data_matches_dvc": "no"})
    assert not feature_store.is_proven(proven | {"data_commit": "not committed"})
    assert feature_store.is_proven(proven | {"data_commit": "not committed", "data_tag": "data-v1"})


def test_a_pointer_with_another_working_folder_is_not_the_feature_table(tmp_path):
    features, pointer, cache, dir_md5 = fake_dvc(tmp_path, b"the stored table")
    # "wdir: archive" moves the output to data/archive/features: a table left in
    # data/features/ is not this version, even with "path: features".
    pointer.write_text(f"wdir: archive\n{pointer_text(dir_md5)}", encoding="utf-8")

    version = feature_store.data_version(features, pointer=pointer, cache=cache)

    assert version["data_matches_dvc"] == "unknown"


def test_only_the_exact_form_dvc_writes_is_accepted():
    dir_md5 = "a" * 32 + ".dir"
    assert feature_store.pointer_md5(pointer_text(dir_md5)) == dir_md5
    # Windows line ends are the same pointer.
    assert feature_store.pointer_md5(pointer_text(dir_md5).replace("\n", "\r\n")) == dir_md5
    # Valid YAML in another form is not accepted: the same key in quotes (found in a
    # review on 24.11.2026), a working folder (also "."), another field, a renamed
    # output, a second output.
    for other_form in (
        f'"wdir": archive\n{pointer_text(dir_md5)}',
        f"wdir: .\n{pointer_text(dir_md5)}",
        pointer_text(dir_md5) + "  cache: false\n",
        pointer_text(dir_md5, path="archive"),
        pointer_text(dir_md5) + pointer_text(dir_md5).replace("outs:\n", ""),
        # A full-width digit is a digit for Python's \d, but not for DVC (review of 25.11.2026).
        pointer_text(dir_md5).replace("nfiles: 1", "nfiles: １"),
    ):
        assert feature_store.pointer_md5(other_form) == ""


def test_a_stored_pointer_is_read_exactly_as_stored(repo, monkeypatch):
    # A commit stores the pointer with a no-break space after "features": another folder
    # name for DVC. Trimming Git's output would make it look like the real pointer.
    repo.pointer.write_text(pointer_text(repo.dir).replace("path: features", "path: features "), encoding="utf-8")
    run_git("commit", "-q", "-am", "a pointer with another folder name")
    monkeypatch.setattr(feature_store, "fetch", lambda commit, out: write_table(out, repo.table_bytes))

    # Asked for by name, that commit has no feature table ...
    with pytest.raises(ValueError, match="has no feature table"):
        feature_store.load_features("HEAD")

    # ... and the checked-out table A (real pointer, not committed) is not given that commit.
    repo.pointer.write_text(pointer_text(repo.dir), encoding="utf-8")
    check_out_table_a(repo)
    _, data = feature_store.load_features()
    assert data["data_commit"] == "not committed"
    assert data["data_tag"] == "data-2026-11-24"


def test_the_real_pointer_of_this_repository_is_accepted():
    # data/features.dvc as the feature pipeline commits it every evening: if DVC ever
    # writes another form, this test shows it before training stops registering models.
    real = Path(__file__).parent.parent / "data" / "features.dvc"

    assert feature_store.pointer_md5(real.read_text(encoding="utf-8")).endswith(".dir")


def test_tag_names_are_cut_only_at_line_feeds(repo):
    # A no-break space at the end of a name, and a Unicode line separator inside one:
    # both are allowed in tag names and must stay part of the name.
    ending = "data-end "
    inside = "data-a ms3"
    run_git("tag", ending)
    run_git("tag", inside)
    run_git("tag", "ms3")  # not a data tag

    tags = feature_store.data_tag(repo.dir).split(",")

    assert ending in tags and inside in tags
    assert "ms3" not in tags and "data-end" not in tags
