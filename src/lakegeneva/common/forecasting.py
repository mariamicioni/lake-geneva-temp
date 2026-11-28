"""From the model's answer to the forecast: the last reading plus the predicted change.

The model does not predict tomorrow's temperature itself. It predicts tomorrow's
CHANGE: tomorrow's mean minus the last reading before 18:00 (train.change). To get a
temperature, the last reading must be added again. This file is the only place where
that step is made. It has two users:

- the walk-forward test of the training pipeline (train.walk_forward), and
- the live forecast of MS4, on the model it loads from the MLflow registry.
"""
# Why one shared file: training-serving skew. Before, "last reading + change" was one
# line inside train.walk_forward, and the registered model
# (models:/lake-geneva-temp@champion) returns only the change, about -0.3 degC. The
# live forecast would have had to write the step a second time by hand (found in a
# review on 15.11.2026). Two copies of one step can drift apart: for example, one
# copy adds today's mean instead of the last reading. Then the live forecast is not
# the forecast that was tested, and no test sees it. With one function, the tested
# forecast and the live forecast are the same code.
#
# Why a function, and not a model that returns the temperature itself:
# 1. All registered versions mean the same thing. Versions 1 and 2 return the change.
#    If newer versions returned the temperature, the alias "champion" could point at
#    either kind (the gate can keep an old version), and the live code would have to
#    know which. A wrong guess is wrong by a whole water temperature, without any
#    error message. This function works with every version, also the old ones.
# 2. A model that adds the last reading itself would need a class of this project.
#    MLflow saves a model with pickle, and pickle saves only the NAME of a class, not
#    its code. So loading such a model would need this project's code anyway, and an
#    old version could not be loaded any more after the class is renamed or moved.
#    The plain scikit-learn pipeline that is registered now loads without our code.
# 3. train.walk_forward trains hundreds of models on plain arrays, because that is
#    fast. A function can be called on arrays directly.
# The weak point of a function: model.predict() alone still gives the change. Three
# things protect against using it as a temperature: the MLflow signature of the
# model names its output MODEL_OUTPUT (track.log_run), the run parameter "target"
# says what the model predicts, and tests/test_track.py loads the champion the way
# MS4 will and checks that the result is a temperature.

from typing import Protocol  # type hint: "anything that has these methods"

import numpy as np  # arrays: the last readings and the predicted changes
import pandas as pd  # the feature table, and the forecasts with their dates

# The name of the model's output in MLflow (the "signature" of the model: its input
# columns and its output). With this name the model page itself says that the number
# is a change in degC, not a temperature.
MODEL_OUTPUT = "change_degC"


class Predictor(Protocol):
    """Anything with predict(): the scikit-learn pipeline of train.make_model(), or a small fake in a test."""

    # A Protocol only describes what the functions below use (review of 27.11.2026: the
    # hint "Pipeline" was narrower than the code, which works with any such model).
    # The "/" makes x positional only: the functions below call predict(x), and
    # scikit-learn names this argument X, so predict(x=...) would not work there.
    def predict(self, x: pd.DataFrame | np.ndarray, /) -> np.ndarray: ...


class TablePredictor(Predictor, Protocol):
    """A Predictor that was trained on a table: scikit-learn then keeps its column names in feature_names_in_."""

    # A property: the functions only read the names, and scikit-learn sets them itself
    # when the model is trained (review of 28.11.2026: a plain attribute would also allow writing).
    @property
    def feature_names_in_(self) -> np.ndarray: ...


def temperature(
    model: Predictor, x: pd.DataFrame | np.ndarray, water_last: pd.Series | np.ndarray | list[float]
) -> np.ndarray:
    """Tomorrow's mean water temperature (°C): the last reading plus the change the model predicts.

    model: a trained model of train.make_model(); model.predict(x) gives the change.
    x: the feature rows, one row per day, with the columns the model was trained on
    (a table or a plain array).
    water_last: the last reading before 18:00 of the same days, in the same order.
    """
    # The reverse of train.change(): there, change = label - water_last; here,
    # forecast = water_last + predicted change.
    # np.asarray(..., dtype=float) makes a plain array of numbers out of whatever came
    # in (an array, a column of a table or a list), so the sum is one number per row.
    return np.asarray(water_last, dtype=float) + model.predict(x)


def temperature_for_rows(model: TablePredictor, rows: pd.DataFrame) -> pd.Series:
    """The forecast for rows of the feature table, one per day: what the live forecast calls.

    model: a model that was trained on a table (train.train_final), for example the
    champion loaded from MLflow.
    rows: rows of the feature table. Columns that the model does not use are ignored.
    """
    # A model that was trained on a table remembers the names and the order of its
    # columns (feature_names_in_, set by scikit-learn). Taking exactly these columns
    # in this order means that the live forecast cannot hand the model other columns
    # than in training (this would be training-serving skew again). A model that was
    # trained on plain arrays (inside train.walk_forward) has no names: for it, use
    # temperature() above.
    columns = list(model.feature_names_in_)
    values = temperature(model, rows[columns], rows["water_last"])
    # The same form as the result of train.walk_forward: one forecast per date.
    return pd.Series(values, index=rows.index, name="forecast")
