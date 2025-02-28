""" Constants used throughout project. """

import enum

RESIDUAL_LABEL = "residual"
CHANGE_PROB_LABEL = "change_prob"
ESTIMATE_LABEL = "estimate"
ESTIMATE_PREDICTED_LABEL = "estimate_predicted"
TIMESTAMP_LABEL = "timestamp"
FRACTION_OF_YEAR_LABEL = "frac_of_year"
DATE_LABEL = "date"
AMPLITUDE_LABEL = "amplitude"
MEASUREMENT_LABEL = "measurement"

PROBABILITY_LABEL = "probability_array"
PROBABILITY_SELECTOR = "probability_class"
PER_PIXEL_IMAGE_COUNTER = "per_pixel_image_counter"
BULC_CONFIDENCE = "BULC_confidence"
BULC_CLASSIFICATION = "BULC_classification"
KALMAN_STATE = "kalman_state"
KALMAN_STATE_COV = "kalman_state_covariance"
GLOBAL_COUNTER = "number_of_images_so_far"
CONDITIONAL_STACK = "conditional_stack"
COMPARISON_LAYER = "comparison_layer"
EVENT = "event"
GLOBAL_COUNTER_PROPERTY_NAME = "number_of_images_so_far"

RECORDING_FLAGS = {
    "iteration_number": False,
    "events": False,
    "conditionals": False,
    "probabilities": True,
    "bulc_layers": True,
    "confidence": False,
    "final_class": True,
    "final_probabilities": True,
    "ccdc_coefficients": True,
    "estimate": True,
    "measurement": True,
    "timestamp": True,
}


class KalmanRecordingFlags(enum.Enum):
    STATE = "state"
    STATE_COV = "state_covariance"
    ESTIMATE = ESTIMATE_LABEL
    ESTIMATE_PREDICTED = ESTIMATE_PREDICTED_LABEL
    TIMESTAMP = TIMESTAMP_LABEL
    FRACTION_OF_YEAR = FRACTION_OF_YEAR_LABEL
    AMPLITUDE = AMPLITUDE_LABEL
    MEASUREMENT = MEASUREMENT_LABEL
    CCDC_COEFFICIENTS = "ccdc_coefficients"


class BulcRecordingFlags(enum.Enum):
    ITERATION_NUMBER = "iteration_number"
    EVENTS = "events"
    CONDITIONALS = "conditionals"
    PROBABILITIES = "probabilities"
    BULC_LAYERS = "bulc_layers"
    CONFIDENCE = "confidence"
    FINAL_CLASS = "final_class"
    FINAL_PROBABILITIES = "final_probabilities"


class Index(enum.Enum):
    SWIR = "swir"
    NBR = "nbr"
    NDVI = "ndvi"


class Sensor(enum.Enum):
    L7 = "L7"
    L8 = "L8"
    L9 = "L9"
    S2 = "S2"


class Kalman(enum.Enum):
    F = "F"  # state transition model
    Q = "Q"  # process noise covariance
    H = "H"  # observation model
    R = "R"  # measurement noise covariance
    P = "P"  # state covariance matrix
    X = "X"  # state
    Z = "z"  # observation
    INITIAL_STATE = "initial_state"
    COV_PREFIX = "cov"
    EOY_STATE = "eoy_state"


FORWARD_TREND_LABEL = "forward_trend"
RETROFITTED_TREND_LABEL = "retrofitted_trend"


class Initialization(enum.Enum):
    UNIFORM = "uniform"
    POSTHOC = "posthoc"
    CCDC = "ccdc"


class Harmonic(enum.Enum):
    INTERCEPT = "INTP"
    SLOPE = "SLP"
    COS = "COS"
    SIN = "SIN"
    COS2 = "COS2"
    SIN2 = "SIN2"
    COS3 = "COS3"
    SIN3 = "SIN3"
    UNIMODAL = "unimodal"
    BIMODAL = "bimodal"
    TRIMODAL = "trimodal"
    FIT = "fit"


class CCDC(enum.Enum):
    BAND_PREFIX = "ccdc"
    FIT = "ccdc_fit"
    SEGMENTS = "segments"


HARMONIC_TAGS = ["INTP", "SLP", "COS", "SIN", "COS2", "SIN2", "COS3", "SIN3"]
HARMONIC_FLAGS_LABEL = "harmonic_flags"
HARMONIC_TREND_LABEL = "harmonic_trend"

NUM_MEASURES = 1  # eeek only supports one band at a time

MASK_VALUE = -999

POINT_INDEX_LABEL = "point_index"


class KalmanModels(enum.Enum):
    UNIMODAL = "unimodal"
    UNIMODAL_WITH_SLOPE = "unimodal_with_slope"
    BIMODAL = "bimodal"
    BIMODAL_WITH_SLOPE = "bimodal_with_slope"
    TRIMODAL = "trimodal"
    TRIMODAL_WITH_SLOPE = "trimodal_with_slope"


KALMAN_MODELS = [
    KalmanModels.UNIMODAL,
    KalmanModels.UNIMODAL_WITH_SLOPE,
    KalmanModels.BIMODAL,
    KalmanModels.BIMODAL_WITH_SLOPE,
    KalmanModels.TRIMODAL,
    KalmanModels.TRIMODAL_WITH_SLOPE,
]

KALMAN_MODEL_HARMONIC_FLAGS = {
    KalmanModels.UNIMODAL.value: {
        Harmonic.INTERCEPT.value: True,
        Harmonic.UNIMODAL.value: True,
    },
    KalmanModels.UNIMODAL_WITH_SLOPE.value: {
        Harmonic.INTERCEPT.value: True,
        Harmonic.SLOPE.value: True,
        Harmonic.UNIMODAL.value: True,
    },
    KalmanModels.BIMODAL.value: {
        Harmonic.INTERCEPT.value: True,
        Harmonic.BIMODAL.value: True,
    },
    KalmanModels.BIMODAL_WITH_SLOPE.value: {
        Harmonic.INTERCEPT.value: True,
        Harmonic.SLOPE.value: True,
        Harmonic.BIMODAL.value: True,
    },
    KalmanModels.TRIMODAL.value: {
        Harmonic.INTERCEPT.value: True,
        Harmonic.TRIMODAL.value: True,
    },
    KalmanModels.TRIMODAL_WITH_SLOPE.value: {
        Harmonic.INTERCEPT.value: True,
        Harmonic.SLOPE.value: True,
        Harmonic.TRIMODAL.value: True,
    },
}
