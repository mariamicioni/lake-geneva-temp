"""One way to the internet for all download scripts, with a replay mode for checks.

Normally open_url() is urllib.request.urlopen(). If the environment variable
LAKEGENEVA_REPLAY names a folder, it reads the saved answer of the same API from
that folder instead, and nothing goes to the internet:

    LAKEGENEVA_REPLAY=tests/fixtures/replay python -m lakegeneva.features.run_pipeline

CI runs the whole Docker image like this on saved answers (MS2 review point A3), so a
change that breaks the downloads or the table inside the image fails in CI, before
the nightly job runs. The saved answers are in tests/fixtures/replay/ (README there).
"""
# Why one file per API and not per request: a check only needs answers in the real
# format. The scripts ask for "the last 7 days" or one run per day, so the dates of the
# requests change every day; the saved answer is the same for all of them.
# Why not a fake web server: the BAFU API needs POST requests, and a file is simpler.

import os  # os.environ: is the replay mode on?
import urllib.parse  # splits a URL into its host and path
import urllib.request  # the real request
from pathlib import Path  # the folder of the saved answers
from typing import IO  # type hint: an open file or a web answer (both can be read)

# The name of the environment variable that turns the replay mode on.
REPLAY = "LAKEGENEVA_REPLAY"


class MissingAnswer(Exception):
    """The replay folder has no saved answer for this API."""

    # Not an OSError on purpose: the download scripts retry OSError three times with
    # a minute of waiting. A missing file will not appear by waiting, so the run stops
    # at once, with the name of the file it looked for.


def answer_name(url: str) -> str:
    """The file name of the saved answer of a URL: its host and path, e.g. data.bafu.admin.ch_api.json."""
    parts = urllib.parse.urlsplit(url)
    # The query (?latitude=...&start_date=...) is left out: it changes every day.
    path = parts.path.strip("/").replace("/", "_")
    return f"{parts.netloc}_{path}.json" if path else f"{parts.netloc}.json"


def open_url(request: urllib.request.Request, timeout: float) -> IO[bytes]:
    """The answer to `request`: from the web, or from the replay folder if LAKEGENEVA_REPLAY is set."""
    folder = os.environ.get(REPLAY)
    if not folder:
        # The normal case: the real request, exactly as before (the tests replace
        # urllib.request.urlopen with a fake, and that still works through here).
        return urllib.request.urlopen(request, timeout=timeout)
    path = Path(folder) / answer_name(request.full_url)
    if not path.is_file():
        raise MissingAnswer(f"replay mode ({REPLAY}={folder}): no saved answer {path.name} for {request.full_url}")
    # "rb": bytes, like a web answer, so json.load() reads both the same way.
    return open(path, "rb")
