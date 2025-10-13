import argparse
import csv
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import ee
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kalman.kalman_module import fetch_ccdc_coefficients
from lib.constants import CCDC, HARMONIC_TAGS

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

LON_MIN = -64.9118
LON_MAX = -63.9533
LAT_MIN = -10.6813
LAT_MAX = -10.1315
AOI = ee.Geometry.Rectangle([LON_MIN, LAT_MIN, LON_MAX, LAT_MAX])

DEFAULT_START_DATE = "2012-01-01"
DEFAULT_END_DATE = "2019-12-31"
DEFAULT_OUTPUT_DIR = Path("kalman/random_pixel_datasets")
DEFAULT_CHUNK_SIZE = 50
DEFAULT_RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_TILE_SCALE = 4

BASE_FIELDNAMES = [
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

CCDC_FIELDNAMES = [f"{CCDC.BAND_PREFIX.value}_{tag}" for tag in HARMONIC_TAGS]
CCDC_FIELDNAMES.append(CCDC.FIT.value)


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
    include_ccdc: bool = True

def mask_landsat_surface_reflectance(image: ee.Image) -> ee.Image:
    # Bit 0 - Fill
    # Bit 1 - Dilated Cloud
    # Bit 2 - Cirrus
    # Bit 3 - Cloud
    # Bit 4 - Cloud Shadow
    qa_mask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    saturation_mask = image.select('QA_RADSAT').eq(0)
    return image.updateMask(qa_mask).updateMask(saturation_mask)


def scale_landsat_surface_reflectance(image: ee.Image) -> ee.Image:
    optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
    thermal = image.select("ST_B.*").multiply(0.00341802).add(149.0)
    return ee.Image(image.addBands(optical, None, True).addBands(thermal, None, True))


def mask_sentinel2_surface_reflectance(image: ee.Image) -> ee.Image:
    params = {
        'QA_Band_Name_Input': 'QA60',
        'cloudBit': 2**10,
        'cirrusBit': 2**11,
        'waterBit': 2**11,
        'clouds_bit_Thresh': 0,
        'cirrus_bit_Thresh': 0,
        'water_bit_Thresh': 0
    }

    input_image = ee.Image(image)
    cloud_bqa = input_image.select(params['QA_Band_Name_Input'])

    cloud_mask = (
        cloud_bqa.bitwiseAnd(params['cloudBit']).eq(params['clouds_bit_Thresh'])
        .And(cloud_bqa.bitwiseAnd(params['cirrusBit']).eq(params['cirrus_bit_Thresh']))
        .And(cloud_bqa.bitwiseAnd(params['waterBit']).eq(params['water_bit_Thresh']))
    )

    return input_image.updateMask(cloud_mask)

def scale_sentinel2_surface_reflectance(image: ee.Image) -> ee.Image:
    optical = image.select("B.*").multiply(0.0001)
    return ee.Image(image.addBands(optical, None, True))

def build_sensor_configs() -> Dict[str, SensorConfig]:
    return {
        "Landsat8": SensorConfig(
            name="Landsat8",
            collection_id="LANDSAT/LC08/C02/T1_L2",
            band_map={
                "SWIR2": "SR_B7",
                "NIR": "SR_B5",
                "RED": "SR_B4"
            },
            scale=30,
            mask_fn=mask_landsat_surface_reflectance,
            scale_fn=scale_landsat_surface_reflectance,
            start_date="2013-03-18",
        ),
        "Sentinel2": SensorConfig(
            name="Sentinel2",
            collection_id="COPERNICUS/S2_SR_HARMONIZED",
            band_map={
                "SWIR2": "B12",
                "NIR": "B8",
                "RED": "B4",
            },
            scale=20,
            mask_fn=mask_sentinel2_surface_reflectance,
            scale_fn=scale_sentinel2_surface_reflectance,
            start_date="2017-03-28",
            include_ccdc=False,
        ),
    }


def chunked(sequence: Sequence[Point], size: int) -> Iterable[Sequence[Point]]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


def generate_random_points(count: int, prefix: str, rng: random.Random) -> List[Point]:
    points: List[Point] = []
    for idx in range(count):
        lon = rng.uniform(LON_MIN, LON_MAX)
        lat = rng.uniform(LAT_MIN, LAT_MAX)
        points.append(Point(f"{prefix}_{idx:05d}", lon, lat))
    return points


def load_points_from_asset(
    asset_id: str,
    id_property: str,
    prefix: str,
    limit: Optional[int],
) -> List[Point]:
    collection = ee.FeatureCollection(asset_id)
    if limit is not None:
        collection = collection.limit(limit)

    info = collection.getInfo()
    features = info.get("features", [])
    points: List[Point] = []

    for idx, feature in enumerate(features):
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [])
        if not coordinates or len(coordinates) < 2:
            continue

        lon, lat = float(coordinates[0]), float(coordinates[1])
        properties = feature.get("properties", {})
        identifier = properties.get(id_property) or feature.get("id")
        if identifier is None:
            identifier = f"{idx:05d}"

        suffix = str(identifier).replace(" ", "_")
        points.append(Point(f"{prefix}_{suffix}", lon, lat))

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
    band_label: str
) -> ee.ImageCollection:

    effective_start = start_date
    if config.start_date and config.start_date > start_date:
        effective_start = config.start_date

    collection = ee.ImageCollection(config.collection_id)
    # .filterBounds(AOI)
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
        value_band = add_time_band(value_band)
        if config.include_ccdc:
            value_band = fetch_ccdc_coefficients(
                value_band,
                ee.Number(value_band.get("system:time_start")),
                [band_label],
            )

        return value_band

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
    include_ccdc: bool,
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

        lookup_point_id = str(point_id)
        point_id = lookup_point_id.split("_", 1)[-1]
        timestamp = int(millis)
        row: Dict[str, object] = {
            "point_id": point_id,
            "set_type": set_type,
            "sensor": sensor_name,
            "band": band_label,
            "longitude": coordinates[0],
            "latitude": coordinates[1],
            "timestamp": timestamp,
            "date": datetime.utcfromtimestamp(timestamp / 1000).strftime("%Y-%m-%d"),
            "observed": float(observed),
            "point_lookup_id": lookup_point_id,
        }

        if include_ccdc:
            for field in CCDC_FIELDNAMES:
                value = properties.get(field)
                row[field] = float(value) if value is not None else None

        rows.append(row)

    rows.sort(key=lambda row: (row["point_id"], row["timestamp"]))
    return rows


def build_fieldnames(include_ccdc: bool) -> List[str]:
    fieldnames = list(BASE_FIELDNAMES)
    if include_ccdc:
        fieldnames.extend(CCDC_FIELDNAMES)
    return fieldnames


def open_dataset_writer(
    output_dir: Path,
    sensor_name: str,
    band_label: str,
    set_type: str,
    set_size: int,
    fieldnames: Sequence[str],
) -> tuple[csv.DictWriter, object]:
    filename = build_output_filename(band_label, sensor_name, set_size, set_type)
    path = output_dir / sensor_name / band_label / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return writer, handle


def close_dataset_writer(handle: object) -> None:
    handle.close()


def stream_samples_to_writer(
    collection: ee.ImageCollection,
    points: Sequence[Point],
    set_type: str,
    sensor_name: str,
    band_label: str,
    scale: int,
    tile_scale: int,
    chunk_size: int,
    set_size: int,
    writer: csv.DictWriter,
    fieldnames: Sequence[str],
    include_ccdc: bool,
) -> None:

    if set_size <= 0 or writer is None:
        return

    selected_points = list(points[:set_size])
    if not selected_points:
        return

    index_lookup = {point.id: idx for idx, point in enumerate(selected_points)}

    for chunk in chunked(selected_points, chunk_size):
        features = sample_collection_for_points(collection, chunk, scale, tile_scale)
        rows = features_to_rows(features, set_type, sensor_name, band_label, include_ccdc)
        if not rows:
            continue

        for row in rows:
            lookup_key = row.get("point_lookup_id", row["point_id"])
            point_idx = index_lookup.get(lookup_key)
            if point_idx is None:
                continue

            row_to_write = {field: row.get(field) for field in fieldnames}
            writer.writerow(row_to_write)


def build_output_filename(
    band_label: str,
    sensor_name: str,
    set_size: int,
    set_type: str,
) -> str:
    capitalized_sensor = sensor_name
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
        "--sample-size",
        type=int,
        help="Number of random points to sample for each dataset when no asset is provided.",
    )
    parser.add_argument(
        "--train-asset",
        help="Earth Engine FeatureCollection asset ID providing training points.",
    )
    parser.add_argument(
        "--test-asset",
        help="Earth Engine FeatureCollection asset ID providing testing points.",
    )
    parser.add_argument(
        "--train-id-property",
        default="system:index",
        help="Feature property to use as the identifier for training asset points.",
    )
    parser.add_argument(
        "--test-id-property",
        default="system:index",
        help="Feature property to use as the identifier for testing asset points.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("Sample size must be positive when provided.")

    rng_train = random.Random(args.seed)
    rng_test = random.Random(args.seed + 1)

    sensors = build_sensor_configs()

    random_sample_size = args.sample_size or DEFAULT_SAMPLE_SIZE
    asset_limit = args.sample_size if args.sample_size and args.sample_size > 0 else None

    if args.train_asset:
        train_points = load_points_from_asset(
            args.train_asset,
            args.train_id_property,
            "train",
            asset_limit,
        )
    else:
        train_points = generate_random_points(random_sample_size, "train", rng_train)

    if args.test_asset:
        test_points = load_points_from_asset(
            args.test_asset,
            args.test_id_property,
            "test",
            asset_limit,
        )
    else:
        test_points = generate_random_points(random_sample_size, "test", rng_test)

    for sensor_name, config in sensors.items():
        fieldnames = build_fieldnames(config.include_ccdc)
        for band_label, band_name in config.band_map.items():
            collection = prepare_collection(config, band_name, args.start_date, args.end_date, band_label)
            collection_size = collection.size().getInfo()
            has_data = collection_size > 0

            for set_type, points in (("train", train_points), ("test", test_points)):
                set_size = len(points)
                if set_size == 0:
                    continue

                writer, handle = open_dataset_writer(
                    args.output_dir,
                    sensor_name,
                    band_label,
                    set_type,
                    set_size,
                    fieldnames,
                )

                try:
                    if has_data:
                        stream_samples_to_writer(
                            collection,
                            points,
                            set_type,
                            sensor_name,
                            band_label,
                            config.scale,
                            args.tile_scale,
                            args.chunk_size,
                            set_size,
                            writer,
                            fieldnames,
                            config.include_ccdc,
                        )
                finally:
                    close_dataset_writer(handle)


if __name__ == "__main__":
    main()
