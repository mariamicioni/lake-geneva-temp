"""Tests for the shared step from the model's answer (a change) to the forecast (a temperature)."""
# How to run: `uv run pytest` (locally) - CI runs the same on every push.
# Why these tests: the model returns tomorrow's CHANGE. The step "last reading + change"
# must be the same code in the walk-forward test and in the live forecast (MS4).
# Otherwise the live forecast is not the forecast that was tested (training-serving skew).
import numpy as np
import pytest  # pytest.approx: compares decimal numbers with a tiny tolerance

from lakegeneva.common import features, forecasting
from lakegeneva.training import train
from tests.test_train import fake_rows


class FixedChange:
    """A fake model: it "predicts" the same change for every row and keeps what it was given."""

    def __init__(self, change: float, columns: list[str] | None = None):
        self.change = change
        if columns is not None:
            # scikit-learn sets this on a model that was trained on a table.
            self.feature_names_in_ = np.array(columns)

    def predict(self, x):
        self.given = x  # the rows the model got, so a test can look at them
        return np.full(len(x), self.change)


def test_the_forecast_is_the_last_reading_plus_the_predicted_change():
    model = FixedChange(-0.3)  # tomorrow is 0.3 degC colder than the last reading
    x = np.zeros((2, 3))  # two days; the fake model does not read the features

    forecast = forecasting.temperature(model, x, water_last=[15.0, 20.0])

    # One temperature per day: the last reading of that day plus the change.
    assert list(forecast) == pytest.approx([14.7, 19.7])
    # Not the change alone: that would be -0.3 on both days.
    assert list(forecast) != pytest.approx([-0.3, -0.3])


def test_rows_of_the_table_get_the_columns_of_the_model_in_its_order():
    # The feature table has more columns than the model uses (here also the label),
    # and this model was trained on two of them, in another order than in the table.
    rows = fake_rows(5)
    model = FixedChange(0.5, columns=["wind_sw_fc_d1", "water_last"])

    forecast = forecasting.temperature_for_rows(model, rows)

    # The model got exactly its own columns in its own order. The label (the truth of
    # tomorrow) was not handed over.
    assert list(model.given.columns) == ["wind_sw_fc_d1", "water_last"]
    # One forecast per date, in the same form as the result of train.walk_forward.
    assert forecast.index.equals(rows.index)
    assert forecast.name == "forecast"
    # The last reading of each day plus the change of 0.5 degC.
    assert list(forecast) == pytest.approx(list(rows["water_last"] + 0.5))


def test_the_final_model_gives_the_same_forecast_as_the_walk_forward_test(monkeypatch):
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 10)
    rows = fake_rows(60)
    day = rows.index[40]
    # The tested forecast: walk_forward trains a model on the rows known at 18:00 on
    # that day and forecasts the day.
    tested = train.walk_forward(rows, features.FEATURES, 1.0, [day])[day]

    # The live way: the final model (train_final) on the SAME training rows, then the
    # shared function on the row of that day, as the live forecast will call it.
    model = train.train_final(train.known_at_18(rows, day), features.FEATURES, 1.0)
    live = forecasting.temperature_for_rows(model, rows.loc[[day]])[day]

    # The same temperature. approx with a very small tolerance: one model is trained
    # on plain arrays and one on a table, so the last digits may differ on another
    # computer (on the laptop of this project they are exactly equal).
    assert live == pytest.approx(tested, abs=1e-9)
    # And it is a temperature, not a change: close to the true mean of tomorrow
    # (about 14 degC here), while the model's own answer is a change of about 0.1 degC.
    truth = rows.at[day, features.LABEL]
    change = model.predict(rows.loc[[day], features.FEATURES])[0]
    assert abs(live - truth) < 1
    assert abs(change - truth) > 5


def test_the_test_and_the_live_forecast_go_through_the_same_function(monkeypatch):
    monkeypatch.setattr(train, "MIN_TRAIN_ROWS", 10)
    rows = fake_rows(40)
    day = rows.index[-1]
    model = train.train_final(train.known_at_18(rows, day), features.FEATURES, 1.0)

    # Replace the shared function by one that always says 99 degC.
    monkeypatch.setattr(forecasting, "temperature", lambda model, x, water_last: np.full(len(x), 99.0))

    # Both ways now say 99 degC. So neither of them has its own copy of the step
    # "last reading + change": a change to the shared function changes both.
    assert train.walk_forward(rows, features.FEATURES, 1.0, [day])[day] == 99.0
    assert forecasting.temperature_for_rows(model, rows.loc[[day]])[day] == 99.0


def test_a_model_that_knows_the_change_gives_back_the_true_temperature():
    # forecasting.temperature() must be the exact reverse of train.change(). If the
    # model's change is the true change of every day, the forecast is the label.
    rows = fake_rows(5)

    class TrueChange:
        def predict(self, x):
            return train.change(rows).to_numpy()

    forecast = forecasting.temperature(TrueChange(), rows[features.FEATURES], rows["water_last"])

    assert list(forecast) == pytest.approx(list(rows[features.LABEL]))
