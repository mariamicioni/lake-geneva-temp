"""The point on the map of all weather downloads, in one place."""
# Why one file: the three weather scripts must ask for the same point. If one of them
# had other coordinates, its forecast could not be compared with the others.

# Geneva, close to the station and the Bains des Pâquis.
LATITUDE = 46.20
LONGITUDE = 6.14

# A second point, on the open lake: the middle of the Grand Lac between Morges and
# Thonon, where the south-west wind blows along the lake and pushes the warm surface
# water away from Geneva (upwelling). Only for the comparison in experiments.py (MS2
# review point B2: "say why one point is enough or add a second one on the lake");
# the model and the daily job use only the Geneva point.
LAKE_LATITUDE = 46.44
LAKE_LONGITUDE = 6.49
# Which grid cell of the weather model stands for the lake point. The API's default
# ("land") looks for a cell on land with a similar height, so for this point it
# returned the weather of a cell on the shore. "nearest" takes the nearest cell,
# which is over the water. Checked on 26.11.2026 for 01.07.2025: at 00 UTC the default
# gave 6.4 km/h from 317 degrees and 32.5 degC at noon, "nearest" 1.1 km/h from
# 162 degrees and 27.6 degC (found in a review on 26.11.2026). Open-Meteo documents
# the option for its Weather Forecast API (cell_selection: land, the default; sea;
# nearest; [16] in README.md); the Previous Runs API page does not list it, but the
# check above shows that this API uses it too.
LAKE_CELL_SELECTION = "nearest"
