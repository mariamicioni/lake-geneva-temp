"""Feature pipeline, after the data commit: tag the new data version data-YYYY-MM-DD.

The day is the issue day D of the version (features.issue_day, printed by
build_features.py), not the date of the run. The tag is put on the one commit that
stores exactly the pointer that this run built, so `load_features("data-YYYY-MM-DD")`
and `git checkout data-YYYY-MM-DD` give back exactly that version.

    python -m lakegeneva.features.tag_data 2026-11-24 --expect <md5>.dir --commit <sha>
"""
# Why (MS2 review point A1): the data versions carried the date of the commit, and
# GitHub often commits after midnight: in all 14 versions from 12.11. to 25.11.2026
# the date was one or two days late. Now each version is named after its issue day D.
# A tag never moves: a model that logged data_tag data-D must always find exactly its
# table again. If a later version has the same issue day (the newest day was not
# complete yet; once in those 14 versions), its commit message names D, but only the
# first version of that day gets the tag.
# Why --expect: the workflow reads the hash of the pointer it just committed BEFORE
# `git pull --rebase`. After the rebase the commit to publish must still store exactly
# that hash. Comparing only HEAD with the folder after the rebase was not enough: if
# the rebase dropped this run's commit, both showed another version, and the tag would
# have named another table (review of 25.11.2026).

import argparse  # reads the issue day and the expected pointer from the command line
import datetime as dt  # checks that the day is a real date

from lakegeneva.common.feature_store import FEATURES_POINTER, git, pointer_md5


def tag_name(day: dt.date) -> str:
    """The tag of the data version with issue day `day`, e.g. data-2026-11-24."""
    return f"data-{day:%Y-%m-%d}"


def stored_pointer(revision: str) -> str:
    """The data version (pointer hash) stored at a commit or tag ("" if none or another form)."""
    return pointer_md5(git("show", f"{revision}^{{commit}}:{FEATURES_POINTER.as_posix()}", strip=False))


def tag(day: dt.date, expected: str, commit: str = "HEAD") -> str:
    """Tag `commit` as the version of `day`: "created", "exists" (same version) or "kept" (another one).

    expected: the pointer hash that this run built; the commit must store exactly it.
    """
    # The commit as a fixed SHA: the tag is made on exactly this commit, not on
    # whatever HEAD is later.
    sha = git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    if not sha:
        raise SystemExit(f"Unknown commit {commit!r}.")
    if not expected or stored_pointer(sha) != expected:
        raise SystemExit(
            f"Commit {sha[:7]} does not store the data version this run built ({expected or 'none'}): no tag."
        )
    name = tag_name(day)
    if git("rev-parse", "--verify", "--quiet", f"refs/tags/{name}"):
        # The tag exists. Same version (e.g. the same run again): nothing to do. Another
        # version: the tag stays where it is, it never moves.
        return "exists" if stored_pointer(f"refs/tags/{name}") == expected else "kept"
    # An annotated tag (like data-v1), with the day in its message, on the fixed SHA.
    git("tag", "-a", name, "-m", f"data version for issue day {day:%d.%m.%Y}", sha)
    if git("rev-parse", "--verify", "--quiet", f"refs/tags/{name}^{{commit}}") != sha:
        raise SystemExit(f"Could not create the tag {name} on {sha[:7]}.")
    return "created"


def main() -> None:
    # Command-line options; __doc__ (the text at the top) becomes the --help text.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("day", type=dt.date.fromisoformat, help="the issue day D, e.g. 2026-11-24")
    parser.add_argument("--expect", required=True, help="the pointer hash this run built, e.g. <md5>.dir")
    parser.add_argument("--commit", default="HEAD", help="the commit to tag (default: HEAD)")
    args = parser.parse_args()

    status = tag(args.day, args.expect, args.commit)
    messages = {
        "created": "created (pushed together with main)",
        "exists": "already on this version, nothing to do",
        "kept": "already on an earlier version with the same issue day; it stays there (tags never move)",
    }
    print(f"tag {tag_name(args.day)}: {messages[status]}")


# Runs main() only when the file is started as a script, not when it is imported
# (the tests import this file).
if __name__ == "__main__":
    main()
