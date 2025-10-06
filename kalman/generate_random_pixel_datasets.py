from __future__ import annotations

import argparse
import csv
from gc import collect
from pprint import pprint
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import ee

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

LON_MIN = -64.9118
LON_MAX = -63.9533
LAT_MIN = -10.6813
LAT_MAX = -10.1315
AOI = ee.Geometry.Rectangle([LON_MIN, LAT_MIN, LON_MAX, LAT_MAX])

SAMPLE_SIZES = [10]
DEFAULT_START_DATE = "2012-01-01"
DEFAULT_END_DATE = "2018-12-31"
DEFAULT_OUTPUT_DIR = Path("kalman/random_pixel_datasets")
DEFAULT_CHUNK_SIZE = 50
DEFAULT_TILE_SCALE = 4
DEFAULT_RANDOM_SEED = 42

CSV_FIELDNAMES = [
    "point_id",
    "set_type",
    "sensor",
    "band",
    "longitude",
    "latitude",
    "timestamp",
    "date",
    "observed",
]


@dataclass(frozen=True)
class Point:
    id: str
    lon: float
    lat: float


@dataclass
class SensorConfig:
    name: str
    collection_id: str
    band_map: Dict[str, str]
    scale: int
    mask_fn: Callable[[ee.Image], ee.Image] | None = None
    scale_fn: Callable[[ee.Image], ee.Image] | None = None
    start_date: str | None = None

def mask_landsat_surface_reflectance(image: ee.Image) -> ee.Image:

    qa_pixel = image.select("QA_PIXEL")
    qa_radsat = image.select("QA_RADSAT")

    clear_mask = (
        qa_pixel.bitwiseAnd(1).eq(0)
        .And(qa_pixel.bitwiseAnd(1 << 1).eq(0))
        .And(qa_pixel.bitwiseAnd(1 << 2).eq(0))
        .And(qa_pixel.bitwiseAnd(1 << 3).eq(0))
        .And(qa_pixel.bitwiseAnd(1 << 4).eq(0))
        .And(qa_pixel.bitwiseAnd(1 << 5).eq(0))
    )

    saturation_mask = qa_radsat.eq(0)

    return ee.Image(image.updateMask(clear_mask).updateMask(saturation_mask))


def scale_landsat_surface_reflectance(image: ee.Image) -> ee.Image:
    optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
    thermal = image.select("ST_B.*").multiply(0.00341802).add(149.0)
    return ee.Image(image.addBands(optical, None, True).addBands(thermal, None, True))


def mask_sentinel2_surface_reflectance(image: ee.Image) -> ee.Image:

    qa60 = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa60.bitwiseAnd(cloud_bit).eq(0).And(qa60.bitwiseAnd(cirrus_bit).eq(0))
    return ee.Image(image.updateMask(mask))


def scale_sentinel2_surface_reflectance(image: ee.Image) -> ee.Image:
    scaled = image.select("B.*").multiply(0.0001)
    return ee.Image(image.addBands(scaled, None, True))


def build_sensor_configs() -> Dict[str, SensorConfig]:
    return {
        "Landsat8": SensorConfig(
            name="Landsat8",
            collection_id="LANDSAT/LC08/C02/T1_L2",
            band_map={
                "SWIR2": "SR_B7",
                # "NIR": "SR_B5",
                # "RED": "SR_B4"
            },
            scale=30,
            mask_fn=mask_landsat_surface_reflectance,
            scale_fn=scale_landsat_surface_reflectance,
            start_date="2013-04-11",
        ),
        # "Landsat9": SensorConfig(
        #     name="Landsat9",
        #     collection_id="LANDSAT/LC09/C02/T1_L2",
        #     band_map={"SWIR2": "SR_B7", "NIR": "SR_B5", "RED": "SR_B4"},
        #     scale=30,
        #     mask_fn=mask_landsat_surface_reflectance,
        #     scale_fn=scale_landsat_surface_reflectance,
        #     start_date="2021-11-01",
        # ),
        # "Sentinel2": SensorConfig(
        #     name="Sentinel2",
        #     collection_id="COPERNICUS/S2_SR_HARMONIZED",
        #     band_map={"SWIR2": "B12", "NIR": "B8", "RED": "B4"},
        #     scale=10,
        #     mask_fn=mask_sentinel2_surface_reflectance,
        #     scale_fn=scale_sentinel2_surface_reflectance,
        #     start_date="2015-06-23",
        # ),
    }


def chunked(sequence: Sequence[Point], size: int) -> Iterable[Sequence[Point]]:
    """Yield successive chunks from a sequence."""

    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


def generate_random_points(count: int, prefix: str, rng: random.Random) -> List[Point]:
    """Sample random points (lon/lat) within the study region."""

    points: List[Point] = []
    for idx in range(count):
        lon = rng.uniform(LON_MIN, LON_MAX)
        lat = rng.uniform(LAT_MIN, LAT_MAX)
        points.append(Point(f"{prefix}_{idx:05d}", lon, lat))
    return points


def add_time_band(image: ee.Image) -> ee.Image:

    millis = ee.Number(image.get("system:time_start"))
    time_band = ee.Image.constant(millis).toInt64().rename("millis")
    return ee.Image(image).addBands(time_band)


def prepare_collection(
    config: SensorConfig,
    band_name: str,
    start_date: str,
    end_date: str,
) -> ee.ImageCollection:

    effective_start = start_date
    if config.start_date and config.start_date > start_date:
        effective_start = config.start_date

    collection = ee.ImageCollection(config.collection_id).filterBounds(AOI)
    collection = collection.filterDate(effective_start, end_date)

    def apply_all(image: ee.Image) -> ee.Image:
        if config.scale_fn is not None:
            image = config.scale_fn(image)
        if config.mask_fn is not None:
            image = config.mask_fn(image)
        value_band = image.select(band_name).rename("observed")
        value_band = ee.Image(value_band).copyProperties(
            image, image.propertyNames()
        )
        return add_time_band(value_band)
        # return value_band

    return collection.map(apply_all).sort("system:time_start")


def points_to_feature_collection(points: Sequence[Point]) -> ee.FeatureCollection:
    features = [
        ee.Feature(ee.Geometry.Point([point.lon, point.lat]), {"point_id": point.id})
        for point in points
    ]
    return ee.FeatureCollection(features)


def sample_collection_for_points(
    collection: ee.ImageCollection,
    points: Sequence[Point],
    scale: int,
    tile_scale: int,
) -> List[dict]:
    if not points:
        return []

    fc_points = points_to_feature_collection(points)

    def sample_image(image: ee.Image) -> ee.FeatureCollection:
        return image.sampleRegions(
            collection=fc_points,
            scale=scale,
            geometries=True,
            tileScale=tile_scale,
        )

    sampled = ee.FeatureCollection(collection.map(sample_image).flatten())
    info = sampled.getInfo()
    return info.get("features", [])


def features_to_rows(
    features: List[dict],
    set_type: str,
    sensor_name: str,
    band_label: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for feature in features:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates")

        point_id = properties.get("point_id")
        millis = properties.get("millis")
        observed = properties.get("observed")

        if not point_id or millis is None or observed is None or coordinates is None:
            continue

        timestamp = int(millis)
        rows.append(
            {
                "point_id": point_id,
                "set_type": set_type,
                "sensor": sensor_name,
                "band": band_label,
                "longitude": coordinates[0],
                "latitude": coordinates[1],
                "timestamp": timestamp,
                "date": datetime.utcfromtimestamp(timestamp / 1000).strftime("%Y-%m-%d"),
                "observed": float(observed),
            }
        )

    rows.sort(key=lambda row: (row["point_id"], row["timestamp"]))
    return rows


def open_dataset_writers(
    output_dir: Path,
    sensor_name: str,
    band_label: str,
    set_type: str,
    set_sizes: Sequence[int],
) -> tuple[Dict[int, csv.DictWriter], Dict[int, object]]:

    writers: Dict[int, csv.DictWriter] = {}
    handles: Dict[int, object] = {}

    for size in set_sizes:
        filename = build_output_filename(band_label, sensor_name, size, set_type)
        path = output_dir / sensor_name / band_label / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", newline="")
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writers[size] = writer
        handles[size] = handle

    return writers, handles


def close_dataset_writers(handles: Dict[int, object]) -> None:
    for handle in handles.values():
        handle.close()


def stream_samples_to_writers(
    collection: ee.ImageCollection,
    points: Sequence[Point],
    set_type: str,
    sensor_name: str,
    band_label: str,
    scale: int,
    tile_scale: int,
    chunk_size: int,
    set_sizes: Sequence[int],
    writers: Dict[int, csv.DictWriter],
) -> None:

    if not writers:
        return

    index_lookup = {point.id: idx for idx, point in enumerate(points)}
    ordered_sizes = sorted(set_sizes)

    for chunk in chunked(points, chunk_size):
        features = sample_collection_for_points(collection, chunk, scale, tile_scale)
        pprint(features)
        rows = features_to_rows(features, set_type, sensor_name, band_label)
        if not rows:
            continue

        for row in rows:
            point_idx = index_lookup.get(row["point_id"])
            if point_idx is None:
                continue

            for size in ordered_sizes:
                if point_idx < size:
                    writers[size].writerow(row)


def build_output_filename(
    band_label: str,
    sensor_name: str,
    set_size: int,
    set_type: str,
) -> str:
    capitalized_sensor = sensor_name.replace("Sentinel", "Sentinel-")
    suffix = "Training" if set_type.lower() == "train" else "Testing"
    return f"ObservedVSCCDC.{set_size}Points.{band_label}.{capitalized_sensor}.{suffix}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random pixel generator")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )
    parser.add_argument(
        "--tile-scale",
        type=int,
        default=DEFAULT_TILE_SCALE,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=SAMPLE_SIZES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng_train = random.Random(args.seed)
    rng_test = random.Random(args.seed + 1)
    sample_sizes = sorted(set(args.sizes))
    max_size = sample_sizes[-1]

    sensors = build_sensor_configs()

    train_points = generate_random_points(max_size, "train", rng_train)
    test_points = generate_random_points(max_size, "test", rng_test)

    for sensor_name, config in sensors.items():
        for band_label, band_name in config.band_map.items():
            collection = prepare_collection(config, band_name, args.start_date, args.end_date)
            collection_size = collection.size().getInfo()
            has_data = collection_size > 0

            for set_type, points in (("train", train_points), ("test", test_points)):
                writers, handles = open_dataset_writers(
                    args.output_dir,
                    sensor_name,
                    band_label,
                    set_type,
                    sample_sizes,
                )

                try:
                    if has_data:
                        stream_samples_to_writers(
                            collection,
                            points,
                            set_type,
                            sensor_name,
                            band_label,
                            config.scale,
                            args.tile_scale,
                            args.chunk_size,
                            sample_sizes,
                            writers,
                        )
                finally:
                    close_dataset_writers(handles)


if __name__ == "__main__":
    main()
