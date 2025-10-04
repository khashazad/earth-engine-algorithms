import os
import shutil
import random
from lib.image_collections import COLLECTIONS
import ee
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from enum import Enum
import sys

from lib.study_areas import PNW, RANDONIA

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)


class Stability(Enum):
    STABLE = "stable"
    UNSTABLE = "unstable"


DEFAULT_MODE = Stability.UNSTABLE

MODE = Stability(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODE

NUMBER_OF_POINTS_IN_EACH_ITERATION = 10

# Server-side sampling parameters (coarser scale speeds up preselection)
CHANGE_PROXY_SCALE_METERS = 30
MAX_SERVER_SIDE_CANDIDATES = 2000

MINIMUM_MEASUREMENT_COUNT = 15

STUDY_AREA = RANDONIA["coords"]
FIRST_YEAR = 2017
LAST_YEAR = 2018

if MODE == Stability.UNSTABLE:
    MEAN_SWIR_THRESHOLD = 0.1
else:
    MEAN_SWIR_THRESHOLD = 0.1

image_collection = COLLECTIONS["Randonia_l8_l9_2017_2018_swir"]

script_directory = os.path.dirname(os.path.abspath(__file__))

if MODE == Stability.UNSTABLE:
    measurement_directory = os.path.join(script_directory, "new", "unstable")
else:
    measurement_directory = os.path.join(script_directory, "new", "stable")

os.makedirs(measurement_directory, exist_ok=True)


def generate_random_points(polygon, num_points, scale=None):
    min_lat = min(polygon, key=lambda x: x[1])[1]
    max_lat = max(polygon, key=lambda x: x[1])[1]
    min_lon = min(polygon, key=lambda x: x[0])[0]
    max_lon = max(polygon, key=lambda x: x[0])[0]

    points = []
    for _ in range(num_points):
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)

        if scale:
            lat += random.uniform(-scale * 0.00001, scale * 0.00001)
            lon += random.uniform(-scale * 0.00001, scale * 0.00001)

        points.append((lon, lat))

    return points


valid_points_counter = 0


def _build_change_proxy_and_candidates():
    study_area_geom = ee.Geometry.Polygon(STUDY_AREA)

    # Ensure we only use dates in each year window
    first_year_start = ee.Date.fromYMD(FIRST_YEAR, 1, 1)
    first_year_end = ee.Date.fromYMD(FIRST_YEAR + 1, 1, 1)
    last_year_start = ee.Date.fromYMD(LAST_YEAR, 1, 1)
    last_year_end = ee.Date.fromYMD(LAST_YEAR + 1, 1, 1)

    first_year_mean = (
        image_collection.filterDate(first_year_start, first_year_end)
        .select("swir")
        .mean()
    )
    last_year_mean = (
        image_collection.filterDate(last_year_start, last_year_end)
        .select("swir")
        .mean()
    )

    # Change magnitude proxy and sufficient observation mask
    change_magnitude = last_year_mean.subtract(first_year_mean).abs()
    count_all = image_collection.select("swir").reduce(ee.Reducer.count())
    sufficient_obs_mask = count_all.gte(MINIMUM_MEASUREMENT_COUNT)

    if MODE == Stability.UNSTABLE:
        candidate_mask = change_magnitude.gte(MEAN_SWIR_THRESHOLD)
    else:
        candidate_mask = change_magnitude.lt(MEAN_SWIR_THRESHOLD)

    masked = change_magnitude.updateMask(candidate_mask).updateMask(sufficient_obs_mask)

    # Sample server-side to get candidate geometries only where mask is true
    sampled = masked.sample(
        region=study_area_geom,
        scale=CHANGE_PROXY_SCALE_METERS,
        numPixels=MAX_SERVER_SIDE_CANDIDATES,
        geometries=True,
        seed=random.randint(0, 1_000_000),
    )

    # Limit to requested number and return as a FeatureCollection
    limited = ee.FeatureCollection(sampled.toList(NUMBER_OF_POINTS_IN_EACH_ITERATION))

    print(f"sampled size: {limited.size().getInfo()}")

    return limited


def _sample_time_series_for_points(geometries_fc):
    def process_image(image):
        img = image

        def sample_and_copy(feature):
            return feature.copyProperties(
                img, img.propertyNames().remove(ee.String("nominalDate"))
            )

        sampled = img.sampleRegions(
            collection=geometries_fc, scale=10, geometries=True
        ).map(sample_and_copy)

        return sampled

    return ee.FeatureCollection(image_collection.map(process_image).flatten())


def _plot_point_timeseries(data: pd.DataFrame):
    fig, axs = plt.subplots(1, 2, figsize=(24, 8))
    data["date"] = pd.to_datetime(data["date"], unit="ms")

    axs[0].scatter(data["date"], data["swir"], label="SWIR", color="red")
    axs[0].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axs[0].tick_params(axis="x", labelsize=8)
    axs[0].set_title(f"{data['longitude'].iloc[0]},{data['latitude'].iloc[0]}")
    axs[0].legend()

    axs[1].scatter(data["date"], data["swir"], label="SWIR", color="blue")
    axs[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axs[1].tick_params(axis="x", labelsize=8)
    axs[1].set_ylim(0, 0.5)
    axs[1].set_title(f"{data['longitude'].iloc[0]},{data['latitude'].iloc[0]}")
    axs[1].legend()

    title = f"({data['longitude'].iloc[0]},{data['latitude'].iloc[0]})"
    print(title)
    fig.savefig(f"{measurement_directory}/{title}.png")


def main():
    global valid_points_counter

    print("test")

    # Server-side select candidate points using change proxy
    candidate_points = _build_change_proxy_and_candidates()

    # Pull just the candidate points (small number)
    geometries = ee.FeatureCollection(candidate_points)

    # Sample full time series only at those points
    measurements_fc = _sample_time_series_for_points(geometries)

    features = measurements_fc.getInfo()["features"]
    measurements = pd.DataFrame(
        [
            {
                "longitude": feature["geometry"]["coordinates"][0],
                "latitude": feature["geometry"]["coordinates"][1],
                "date": feature["properties"].get("millis"),
                "swir": feature["properties"].get("swir"),
            }
            for feature in features
            if "swir" in feature["properties"]
            and feature["properties"].get("millis") is not None
        ]
    )

    grouped_measurements = measurements.groupby(["longitude", "latitude"])

    # print(grouped_measurements)

    for point, data in grouped_measurements:
        if valid_points_counter >= NUMBER_OF_POINTS_IN_EACH_ITERATION:
            break

        if len(data) < MINIMUM_MEASUREMENT_COUNT:
            continue

        data["year"] = pd.to_datetime(data["date"], unit="ms").dt.year
        mean_swir_first_year = data[data["year"] == FIRST_YEAR]["swir"].mean()
        mean_swir_last_year = data[data["year"] == LAST_YEAR]["swir"].mean()
        difference_in_mean_swir = abs(mean_swir_first_year - mean_swir_last_year)

        print(difference_in_mean_swir)

        if MODE == Stability.UNSTABLE:
            if difference_in_mean_swir < MEAN_SWIR_THRESHOLD:
                continue
        else:
            if difference_in_mean_swir > MEAN_SWIR_THRESHOLD:
                continue

        valid_points_counter += 1
        _plot_point_timeseries(data)


if __name__ == "__main__":
    main()
