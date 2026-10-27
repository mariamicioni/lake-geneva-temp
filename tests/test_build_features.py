"""Tests for the last step of the feature pipeline (raw files -> Parquet)."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
from pathlib import Path

import pandas as pd

from lakegeneva.common import features
from lakegeneva.features import build_features

FIXTURES = Path(__file__).parent / "fixtures"


def test_parquet_file_gives_back_exactly_the_same_table(tmp_path):
    # One raw folder: the real water sample and a fake 48-hour-old forecast.
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "water_2606_sample.csv").write_text((FIXTURES / "water_2606_sample.csv").read_text())
    hours = pd.DataFrame(
        {
            "fetched_at_utc": "2026-10-27T19:00:00+00:00",
            "time_local": pd.date_range("2026-03-27", periods=5 * 24, freq="h").strftime("%Y-%m-%dT%H:%M"),
            features.TEMPERATURE: 8.0,
            features.SUNSHINE: 1800.0,
            features.WIND_SPEED: 12.0,
            features.WIND_DIRECTION: 225.0,
        }
    )
    hours.to_csv(raw / "weather_geneva_prev2_2026-03-27_to_2026-03-31.csv", index=False)
    out = tmp_path / "features" / "features.parquet"  # the folder does not exist yet

    written = build_features.build(raw, out)
    read_back = pd.read_parquet(out)

    # The file exists, and reading it gives exactly the table that was written:
    # the same days (as dates), the same columns, the same values and types.
    assert out.exists()
    pd.testing.assert_frame_equal(read_back, written, check_freq=False)
    # 30.03 has every feature (its trend needs 27.03, its forecast is for 31.03);
    # its label is empty, because the sample ends on 30.03.
    assert read_back.loc[pd.Timestamp("2026-03-30"), features.FEATURES].notna().all()
    assert pd.isna(read_back.loc[pd.Timestamp("2026-03-30"), features.LABEL])
