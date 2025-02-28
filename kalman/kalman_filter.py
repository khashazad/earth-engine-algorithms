import ee
import ee.geometry
from pprint import pprint
from lib import constants
from lib.constants import (
    ESTIMATE_PREDICTED_LABEL,
    Kalman,
    ESTIMATE_LABEL,
    TIMESTAMP_LABEL,
    FRACTION_OF_YEAR_LABEL,
)
from lib.utils.ee.dates import convert_date

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)

# use 2pi * time since this date as input to sinusoids
START_DATE = "2016-01-01"

FREQUENCY = ee.Number(6.283)


def predict(x, P, F, Q):
    """Performs the predict step of the Kalman Filter loop.

    Args:
        x: ee array Image (n x 1), the state
        P: ee array Image (n x n), the state covariance
        F: ee array_image (n x n), the process model
        Q: ee array Image (n x n), the process noise

    Returns:
        x_bar (ee array image), P_bar (ee array image): the predicted state,
        and the predicted state covariance.
    """
    x_bar = F.matrixMultiply(x)
    P_bar = F.matrixMultiply(P).matrixMultiply(F.matrixTranspose()).add(Q)

    return x_bar, P_bar


def update(x_bar, P_bar, z, H, R, num_params):
    """Performs the update step of the Kalman Filter loop.

    Args:
        x_bar: ee array image (n x 1), the predicted state
        P_bar: ee array image (n x n), the predicted state covariance
        z: ee array Image (1 x 1), the measurement
        H: ee array image (1 x n), the measurement function
        R: ee array Image (1 x 1), the measurement noise
        num_params: int, the number of parameters in the state variable

    Returns:
        x (ee array image), P (ee array image): the updated state and state
        covariance
    """
    identity = ee.Image(ee.Array.identity(num_params))

    y = z.subtract(H.matrixMultiply(x_bar))
    S = H.matrixMultiply(P_bar).matrixMultiply(H.matrixTranspose()).add(R)
    S_inv = S.matrixInverse()
    K = P_bar.matrixMultiply(H.matrixTranspose()).matrixMultiply(S_inv)
    x = x_bar.add(K.matrixMultiply(y))
    P = (identity.subtract(K.matrixMultiply(H))).matrixMultiply(P_bar)
    return x, P


def kalman_filter(
    collection,
    init_image,
    F,
    Q,
    H,
    R,
    preprocess_fn=lambda **kwargs: [],
    postprocess_fn=lambda **kwargs: [],
    measurement_band=None,
    num_params=3,
):
    """Applies a Kalman Filter to the given image collection.

    F, Q, H, R, preprocess_fn, and postprocess_fn are all called with
    **locals() as input at each step of the Kalman Filter loop. This gives them
    access to all local variables at the time they are called, in case they
    don't need some/all of those set **kwargs as a parameter in their function
    definition. E.g., if H relies only on t, define it as `def H(t, **kwargs):
    ...` and if Q does not need any inputs, define it as `def Q(**kwargs): ...`

    Args:
        collection: ee.ImageCollection, used as the measurements
        init_image, ee.Image, the initial state, must at least have one band
            containing the guess of the initial state named "x" and one band
            containing the guess of the initial state covariance named "P". The
            user may add other bands to needed by preprocess_fn or
            postprocess_fn in the first iterate step.
        F: function: dict -> ee.Image, the process model
        Q: function: dict -> ee.Image, the process noise
        H: function: dict -> ee.Image, the measurement function
        R: function: dict -> ee.Image, the measurement noise
        preprocess_fn: function: dict -> list[ee.Image], arbitrary function to
            apply at each step of the iterate any images returned by it will
            be stored in the output of each iterate step, triggers before the
            kalman filter predict/update, useful e.g., to run BULC
            simultaneously. Defaults to do nothing.
        postprocess_fn: function dict -> list[ee.Image], arbitrary function to
            apply at each step of the iterate, any images returned by it will
            be stored in the output of each iterate step, triggers AFTER the
            kalman filter predict/update, useful e.g., to track the result of
            evaluating the current state after each update. Default to do
            nothing.
        measurement_band: str, band name for the measurement, if None, the
            first band is used.
        num_params: int, number of parameters in the state, used to create an
            identity matrix with the proper size.

    Returns:
        ee.ImageCollection, the result of applying a Kalman Filter to the input
        collection. Each image in the collection will have one band containing
        the measurement (named z), one band containing an array image of the
        covariance (named P), one band containing an array image of the state
        (named x), and any bands added by either preprocess_fn or
        postprocess_fn. utils.unpack_arrays() can be mapped across this
        ImageCollection to get the state and covariance in a more readable
        form.
    """

    if measurement_band is None:
        measurement_band = collection.first().bandNames().getString(0)

    def _iterator(curr, prev):
        """Kalman Filter Loop."""
        curr = ee.Image(curr)
        prev = ee.List(prev)

        last = ee.Image(prev.get(-1))

        x_prev = last.select(Kalman.X.value)
        P_prev = last.select(Kalman.P.value)

        z = curr.select(measurement_band).toArray().toArray(1)

        year = ee.Date(curr.date().millis()).get("year")
        frac = ee.Date(curr.date().millis()).getFraction("year")

        t = year.add(frac)

        # t = ee.Number(curr.date().getRelative("day", "year")).divide(365)

        preprocess_results = preprocess_fn(**locals())

        x_bar, P_bar = predict(x_prev, P_prev, F(**locals()), Q(**locals()))
        x, P = update(x_bar, P_bar, z, H(**locals()), R(**locals()), num_params)

        postprocess_results = postprocess_fn(**locals())

        x = x_prev.where(curr.select(measurement_band).gt(constants.MASK_VALUE), x)
        P = P_prev.where(curr.select(measurement_band).gt(constants.MASK_VALUE), P)

        estimate = H(**locals()).matrixMultiply(x)
        estimate_predicted = H(**locals()).matrixMultiply(x_bar)
        # amplitude = cos.pow(2).add(sin.pow(2)).sqrt()

        frac_of_year = convert_date(
            {"input_format": 2, "input_date": curr.date(), "output_format": 1}
        )

        outputs = [
            z.rename(Kalman.Z.value),
            x.rename(Kalman.X.value),
            P.rename(Kalman.P.value),
            estimate.rename(ESTIMATE_LABEL),
            estimate_predicted.rename(ESTIMATE_PREDICTED_LABEL),
            # amplitude.rename(constants.AMPLITUDE),
            ee.Image(curr.date().millis()).rename(TIMESTAMP_LABEL),
            ee.Image(frac_of_year).rename(FRACTION_OF_YEAR_LABEL),
        ]
        outputs.extend(preprocess_results)
        outputs.extend(postprocess_results)

        return ee.List(prev).add(
            ee.Image.cat(*outputs).copyProperties(curr, ["system:time_start"])
        )

    result = ee.List(collection.iterate(_iterator, [init_image]))

    # slice to drop the initial image
    return ee.ImageCollection(result.slice(1))
