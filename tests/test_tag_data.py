"""Tests for the data tags data-YYYY-MM-DD (MS2 review point A1). Small Git repositories, no download."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
# The repository comes from test_feature_store.py: one stored data version (table A)
# with the tag data-2026-11-24; its DVC list is added with put_list when needed.
import datetime as dt
import subprocess

import pandas as pd
import pytest

from lakegeneva.common import feature_store
from lakegeneva.features import tag_data
from tests.test_feature_store import TABLE_A, OTHER, pointer_text, put_list, repo, run_git, write_table  # noqa: F401

DAY = dt.date(2026, 11, 25)
THIRD = "e" * 32 + ".dir"  # the hash of a third data version


def store_version(repo, dir_md5: str) -> str:
    """Commit a new data version (its pointer) like the daily job; return the commit."""
    repo.pointer.write_text(pointer_text(dir_md5), encoding="utf-8")
    run_git("commit", "-q", "-am", f"chore(data): daily update {DAY}")
    return run_git("rev-parse", "HEAD")


def test_the_tag_is_named_after_the_issue_day():
    assert tag_data.tag_name(DAY) == "data-2026-11-25"


def test_a_new_version_gets_its_tag_on_the_commit_of_its_pointer(repo):
    commit = store_version(repo, OTHER)

    assert tag_data.tag(DAY, OTHER) == "created"
    # The tag (annotated) stands for exactly that commit.
    assert run_git("rev-parse", "data-2026-11-25^{commit}") == commit


def test_the_same_run_again_leaves_the_tag_in_place(repo):
    store_version(repo, OTHER)
    tag_data.tag(DAY, OTHER)
    tag_object = run_git("rev-parse", "data-2026-11-25")

    # Nothing new: the same version is tagged again.
    assert tag_data.tag(DAY, OTHER) == "exists"
    assert run_git("rev-parse", "data-2026-11-25") == tag_object


def test_a_later_version_with_the_same_issue_day_does_not_move_the_tag(repo):
    first = store_version(repo, OTHER)
    tag_data.tag(DAY, OTHER)
    # A later version with the same issue day (the newest day was not complete yet).
    store_version(repo, THIRD)

    assert tag_data.tag(DAY, THIRD) == "kept"
    # The tag still gives back the first version: a model that logged it finds its table.
    assert run_git("rev-parse", "data-2026-11-25^{commit}") == first


def test_no_tag_if_the_commit_does_not_store_the_version_this_run_built(repo):
    # This run built OTHER, but after `git pull --rebase` HEAD is another version (the
    # rebase dropped this run's commit; review of 25.11.2026). HEAD and the folder agree
    # with each other, but not with what was built: no tag.
    store_version(repo, THIRD)

    with pytest.raises(SystemExit, match="does not store the data version this run built"):
        tag_data.tag(DAY, OTHER)
    assert not run_git("tag", "--list", "data-2026-11-25")


def test_the_tag_goes_on_the_explicit_commit_not_on_head(repo):
    built = store_version(repo, OTHER)
    store_version(repo, THIRD)  # HEAD moved on

    assert tag_data.tag(DAY, OTHER, commit=built) == "created"
    assert run_git("rev-parse", "data-2026-11-25^{commit}") == built


def test_a_refused_tag_push_leaves_main_unchanged_and_a_retry_publishes_both(repo, tmp_path):
    # The push command of the workflow: main and the tag together, atomically.
    remote = tmp_path / "remote.git"
    run_git("init", "-q", "--bare", str(remote))
    run_git("remote", "add", "origin", str(remote))
    run_git("push", "-q", "origin", "HEAD:refs/heads/main")
    before = run_git("ls-remote", "origin", "refs/heads/main").split()[0]
    # The remote refuses the tag: it already has a tag with that name on another commit.
    run_git("push", "-q", "origin", f"{before}:refs/tags/data-2026-11-25")
    store_version(repo, OTHER)
    tag_data.tag(DAY, OTHER)
    push = ["git", "push", "--atomic", "origin", "HEAD:main", "refs/tags/data-2026-11-25"]

    refused = subprocess.run(push, capture_output=True, text=True)

    assert refused.returncode != 0
    # Both or neither: main on the remote is still the old commit.
    assert run_git("ls-remote", "origin", "refs/heads/main").split()[0] == before
    # The cause is removed, and the same push is tried again: now both arrive.
    run_git("push", "-q", "origin", ":refs/tags/data-2026-11-25")
    assert subprocess.run(push, capture_output=True, text=True).returncode == 0
    head = run_git("rev-parse", "HEAD")
    assert run_git("ls-remote", "origin", "refs/heads/main").split()[0] == head
    assert run_git("ls-remote", "origin", "refs/tags/data-2026-11-25^{}").split()[0] == head


def test_the_tagged_version_can_be_loaded_by_its_tag(repo, monkeypatch):
    # Another version is stored, then table A again on 25.11 (its DVC list is in the
    # cache, as after dvc pull) and tagged; load_features by the tag gives back table A
    # and names the tag.
    store_version(repo, OTHER)
    commit = store_version(repo, repo.dir)
    put_list(repo.cache, repo.dir, repo.listing)
    assert tag_data.tag(DAY, repo.dir) == "created"
    fetched = []

    def fake_fetch(commit, out):
        fetched.append(commit)
        write_table(out, repo.table_bytes)

    monkeypatch.setattr(feature_store, "fetch", fake_fetch)

    table, data = feature_store.load_features("data-2026-11-25")

    pd.testing.assert_frame_equal(table, TABLE_A)
    assert fetched == [commit]
    assert "data-2026-11-25" in data["data_tag"].split(",")
    assert feature_store.is_proven(data)
