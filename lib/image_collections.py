"""Code to generate arbitrary image collections.

Use this to change the specific parameters of the image collection that is used
as input to PEST.

Collections defined in this file should NOT have filterBounds or filterDate
applied them. Those filters will be applied inside pest_eeek.py to allow each
point defined in the pest points file to have a different date range and be in
a different location. eeek will run the kalman filter over the first band of
each image in the input image collection so you should always select the band
you want to use when defining the collection.

Ensure that each image collection you define in this file is set as a value in
the COLLECTIONS dictionary.
To set which image collection is used in pest set the value of the
`--collection` flag to the key of the collection in COLLECTIONS that you want
to use for the given run.
"""

import ee
from lib.constants import Index, Sensor
from lib.study_areas import PNW, RANDONIA

ee.Initialize()

from pprint import pprint

from lib.utils.ee.gather_collections import gather_collections_and_reduce


def build_collection(
    study_area,
    years=[2022, 2023],
    index=Index.SWIR,
    sensors=[Sensor.L8, Sensor.L9],
    day_step_size=6,
    start_doy=1,
    end_doy=365,
    cloud_cover_threshold=20,
):
    args = {
        "default_study_area": study_area,
        "band_name_reduction": index.value,
        "which_reduction": index.value.upper(),
        "day_step_size": day_step_size,
        "verbose": False,
        "dataset_selection": {sensor.value: True for sensor in sensors},
        "first_expectation_year": years[0],
    }

    for sensor in sensors:
        args[f"{sensor.value}dictionary"] = {
            "years_list": years,
            "first_doy": start_doy,
            "last_doy": end_doy,
            "cloud_cover_threshold": cloud_cover_threshold,
        }

    return gather_collections_and_reduce(args)


def scale_landsat8(image):
    """Based on https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2"""
    optical_bands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal_bands = image.select("ST_B.*").multiply(0.00341802).add(149.0)
    return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)


L8 = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    # .filter(ee.Filter.lte("CLOUD_COVER", 30))
    .map(scale_landsat8).select("SR_B6")
)

EXPORTED_LANDSAT8_FROM_JS = ee.ImageCollection(
    "projects/api-project-269347469410/assets/kalman-value-collection-60-180"
)

L8_GATHER_COLLECTIONS = gather_collections_and_reduce(
    {
        "L8dictionary": {
            "years_list": [2020],
            "first_doy": 1,
            "last_doy": 365,
            "cloud_cover_threshold": 30,
        },
        "default_study_area": (
            ee.Geometry.Polygon(
                [
                    (-63.9533, -10.6813),
                    (-63.9533, -10.1315),
                    (-64.9118, -10.1315),
                    (-64.9118, -10.6813),
                ]
            )
        ),
        "band_name_reduction": "swir",
        "which_reduction": "SWIR",
        "day_step_size": 4,
        "verbose": False,
        "dataset_selection": {
            "L5": False,
            "L7": False,
            "L8": True,
            "L9": False,
            "MO": False,
            "S2": False,
            "S1": False,
            "DW": False,
        },
        "first_expectation_year": 2020,
        "verbose": False,
    }
)

PNW_L8_L9_2017_2018 = gather_collections_and_reduce(
    {
        "L8dictionary": {
            "years_list": [2017, 2018],
            "first_doy": 1,
            "last_doy": 365,
            "cloud_cover_threshold": 20,
        },
        "L9dictionary": {
            "years_list": [2017, 2018],
            "first_doy": 1,
            "last_doy": 365,
            "cloud_cover_threshold": 20,
        },
        "default_study_area": (ee.Geometry.Polygon(PNW["coords"])),
        "band_name_reduction": "swir",
        "which_reduction": "SWIR",
        "day_step_size": 14,
        "verbose": False,
        "dataset_selection": {
            "L5": False,
            "L7": False,
            "L8": True,
            "L9": True,
            "MO": False,
            "S2": False,
            "S1": False,
            "DW": False,
        },
        "first_expectation_year": 2017,
        "verbose": False,
    }
)

PNW_L8_L9_2022_2023 = gather_collections_and_reduce(
    {
        "L8dictionary": {
            "years_list": [2022, 2023],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 20,
        },
        "L9dictionary": {
            "years_list": [2022, 2023],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 20,
        },
        "default_study_area": (ee.Geometry.Polygon(PNW["coords"])),
        "band_name_reduction": "swir",
        "which_reduction": "SWIR",
        "day_step_size": 6,
        "verbose": False,
        "dataset_selection": {
            "L5": False,
            "L7": False,
            "L8": True,
            "L9": True,
            "MO": False,
            "S2": False,
            "S1": False,
            "DW": False,
        },
        "first_expectation_year": 2022,
        "verbose": False,
    }
)

PNW_L8_L9_2022_2023_DSS_1 = gather_collections_and_reduce(
    {
        "L8dictionary": {
            "years_list": [2022, 2023],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 20,
        },
        "L9dictionary": {
            "years_list": [2022, 2023],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 20,
        },
        "default_study_area": ee.Geometry.Polygon(PNW["coords"]),
        "band_name_reduction": "swir",
        "which_reduction": "SWIR",
        "day_step_size": 1,
        "verbose": False,
        "dataset_selection": {
            "L5": False,
            "L7": False,
            "L8": True,
            "L9": True,
            "MO": False,
            "S2": False,
            "S1": False,
            "DW": False,
        },
        "first_expectation_year": 2022,
        "verbose": False,
    }
)

PNW_L8_L9_2022 = gather_collections_and_reduce(
    {
        "L8dictionary": {
            "years_list": [2022],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 20,
        },
        "L9dictionary": {
            "years_list": [2022],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 20,
        },
        "default_study_area": (ee.Geometry.Polygon(PNW["coords"])),
        "band_name_reduction": "swir",
        "which_reduction": "SWIR",
        "day_step_size": 4,
        "verbose": False,
        "dataset_selection": {
            "L5": False,
            "L7": False,
            "L8": True,
            "L9": True,
            "MO": False,
            "S2": False,
            "S1": False,
            "DW": False,
        },
        "first_expectation_year": 2022,
        "verbose": False,
    }
)

PNW_L8_L9_2023 = gather_collections_and_reduce(
    {
        "L8dictionary": {
            "years_list": [2023],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 20,
        },
        "L9dictionary": {
            "years_list": [2023],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 20,
        },
        "default_study_area": (
            ee.Geometry.Polygon(
                [(-126.04, 49.59), (-126.04, 40.76), (-118.93, 40.76), (-118.93, 49.59)]
            )
        ),
        "band_name_reduction": "swir",
        "which_reduction": "SWIR",
        "day_step_size": 4,
        "verbose": False,
        "dataset_selection": {
            "L5": False,
            "L7": False,
            "L8": True,
            "L9": True,
            "MO": False,
            "S2": False,
            "S1": False,
            "DW": False,
        },
        "first_expectation_year": 2023,
        "verbose": False,
    }
)

CCDC_RANDONIA = ee.Image("projects/GLANCE/RESULTS/CHANGEDETECTION/SA/Rondonia_example")

L8_L9_RANDONIA_SWIR_2017_2018 = gather_collections_and_reduce(
    {
        "L8dictionary": {
            "years_list": [2017, 2018],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 20,
        },
        "L9dictionary": {
            "years_list": [2017, 2018],
            "first_doy": 150,
            "last_doy": 250,
            "cloud_cover_threshold": 20,
        },
        "default_study_area": (ee.Geometry.Polygon(RANDONIA["coords"])),
        "band_name_reduction": "swir",
        "which_reduction": "SWIR",
        "day_step_size": 6,
        "verbose": False,
        "dataset_selection": {
            "L5": False,
            "L7": False,
            "L8": True,
            "L9": True,
            "MO": False,
            "S2": False,
            "S1": False,
            "DW": False,
        },
        "first_expectation_year": 2017,
        "verbose": False,
    }
)

L7_L8_L9_PNW_2017_2018 = gather_collections_and_reduce(
    {
        "L7dictionary": {
            "years_list": [2017, 2018],
            "first_doy": 1,
            "last_doy": 365,
            "cloud_cover_threshold": 20,
        },
        "L8dictionary": {
            "years_list": [2017, 2018],
            "first_doy": 1,
            "last_doy": 365,
            "cloud_cover_threshold": 20,
        },
        "L9dictionary": {
            "years_list": [2017, 2018],
            "first_doy": 1,
            "last_doy": 365,
            "cloud_cover_threshold": 20,
        },
        "default_study_area": (ee.Geometry.Polygon(PNW["coords"])),
        "band_name_reduction": "swir",
        "which_reduction": "SWIR",
        "day_step_size": 4,
        "verbose": False,
        "dataset_selection": {
            "L5": False,
            "L7": True,
            "L8": True,
            "L9": True,
            "MO": False,
            "S2": False,
            "S1": False,
            "DW": False,
        },
        "first_expectation_year": 2017,
        "verbose": False,
    }
)


CCDC_GLOBAL = ee.ImageCollection("GOOGLE/GLOBAL_CCDC/V1")

COLLECTIONS = {
    "PNW_L8_L9_2017_2018": PNW_L8_L9_2017_2018,
    "PNW_L8_L9_2022_2023": PNW_L8_L9_2022_2023,
    "PNW_L8_L9_2022_2023_DSS_1": PNW_L8_L9_2022_2023_DSS_1,
    "PNW_L8_L9_2022": PNW_L8_L9_2022,
    "PNW_L8_L9_2023": PNW_L8_L9_2023,
    "PNW_L7_L8_L9_2017_2018": L7_L8_L9_PNW_2017_2018,
    "CCDC_Randonia": CCDC_RANDONIA,
    "Randonia_l8_l9_2017_2018_swir": L8_L9_RANDONIA_SWIR_2017_2018,
    "CCDC_Global": CCDC_GLOBAL,
}


if __name__ == "__main__":
    pprint(L8_GATHER_COLLECTIONS.getInfo())
