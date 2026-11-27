# Saved API answers for the replay mode

One real answer of each API that the daily feature pipeline calls, recorded on 27.11.2026 during a normal daily run (`run_pipeline`) for the last 7 days. `src/lakegeneva/common/web.py` reads them instead of the internet when `LAKEGENEVA_REPLAY` names this folder (MS2 review point A3).

| File | API | What it holds |
|---|---|---|
| `data.bafu.admin.ch_api.json` | BAFU Data Platform (GraphQL), station 2606 | water temperature and flow, every 10 minutes, 20.–27.11.2026 |
| `previous-runs-api.open-meteo.com_v1_forecast.json` | Open-Meteo Previous Runs API, Geneva | the 48-hour-old forecast, 20.–28.11.2026 |
| `api.open-meteo.com_v1_forecast.json` | Open-Meteo Forecast API, Geneva | the newest forecast (audit copy) |
| `single-runs-api.open-meteo.com_v1_forecast.json` | Open-Meteo Single Runs API, Geneva | one 06:00 UTC run of ECMWF IFS |

The file name is the host and path of the request; the query (dates, point) is left out, so every request to the same API gets the same answer. The scripts ask for one 06:00 UTC run per day, so in replay all 7 runs are the same run: fine for a check that the image works, not for numbers. Public data, no key, nothing personal.

To record them again (for example after an API changed its format): run the daily pipeline once in a temporary folder, save the first answer of each API under `web.answer_name(url)`, and copy the files here.

Used by `tests/test_replay.py` and by the image job of `.github/workflows/ci.yml`. Both check the table built from these answers with the same script, `check_table.py`: issue day 26.11.2026, all ten features on its row, and some rows with a label.
