import ee

from kalman import kalman_filter
from kalman.kalman_helper import (
    parse_harmonic_params,
    parse_band_names,
    setup_kalman_init,
    unpack_kalman_results,
)
from lib.constants import (
    CCDC,
    HARMONIC_TAGS,
    Harmonic,
    Kalman,
    KalmanRecordingFlags,
)
from lib.image_collections import COLLECTIONS
from lib.utils.ee.ccdc_utils import (
    build_ccd_image,
    get_multi_coefs,
    build_segment_tag,
    get_multi_synthetic,
)
from lib.utils.ee.dates import convert_date

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)


def fetch_ccdc_coefficients(image, date, bands):
    ccdc_asset = COLLECTIONS["CCDC_Global"].mosaic()

    coefs = HARMONIC_TAGS

    segments_count = 10
    segments = build_segment_tag(segments_count)

    ccdc_image = build_ccd_image(ccdc_asset, segments_count, bands)

    date = convert_date(
        {
            "input_format": 2,
            "input_date": date,
            "output_format": 1,
        }
    )

    coefs = get_multi_coefs(
        ccdc_image,
        date,
        bands,
        coef_list=HARMONIC_TAGS,
        cond=True,
        segment_names=segments,
        behavior="before",
    ).rename([*[f"{CCDC.BAND_PREFIX.value}_{x}" for x in HARMONIC_TAGS]])

    synthetic_image = get_multi_synthetic(
        ccdc_image,
        date,
        date_format=1,
        band_list=bands,
        segments=segments,
    ).rename(CCDC.FIT.value)

    return image.addBands(coefs.addBands(synthetic_image), overwrite=True)


def append_ccdc_coefficients(image):
    ccdc_asset = COLLECTIONS["CCDC_Global"].mosaic()

    bands = ["SWIR1"]
    coefs = HARMONIC_TAGS

    segments_count = 10
    segments = build_segment_tag(segments_count)

    ccdc_image = build_ccd_image(ccdc_asset, segments_count, bands)

    date = convert_date(
        {
            "input_format": 2,
            "input_date": image.date().millis(),
            "output_format": 1,
        }
    )

    coefs = get_multi_coefs(
        ccdc_image,
        date,
        bands,
        coef_list=HARMONIC_TAGS,
        cond=True,
        segment_names=segments,
        behavior="before",
    ).rename([*[f"{CCDC.BAND_PREFIX.value}_{x}" for x in HARMONIC_TAGS]])

    synthetic_image = get_multi_synthetic(
        ccdc_image,
        date,
        date_format=1,
        band_list=bands,
        segments=segments,
    ).rename(CCDC.FIT.value)

    return image.addBands(coefs.addBands(synthetic_image), overwrite=True)


def main(
    kalman_parameters,
    value_collection,
    harmonic_flags: dict[Harmonic, any],
    recording_flags: dict[KalmanRecordingFlags, bool],
):
    band_names = parse_band_names(recording_flags, harmonic_flags)
    harmonic_params, _ = parse_harmonic_params(harmonic_flags)

    kalman_init = setup_kalman_init(kalman_parameters, harmonic_flags)

    kalman_result = kalman_filter.kalman_filter(
        collection=value_collection,
        init_image=kalman_init.get(Kalman.INITIAL_STATE.value),
        F=kalman_init.get(Kalman.F.value),
        Q=kalman_init.get(Kalman.Q.value),
        H=kalman_init.get(Kalman.H.value),
        R=kalman_init.get(Kalman.R.value),
        num_params=len(harmonic_params),
    )

    states = kalman_result.map(
        lambda im: unpack_kalman_results(im, harmonic_params, recording_flags)
    )

    if recording_flags.get(KalmanRecordingFlags.CCDC_COEFFICIENTS, False):
        states = states.map(append_ccdc_coefficients)

    return states.select(band_names)
