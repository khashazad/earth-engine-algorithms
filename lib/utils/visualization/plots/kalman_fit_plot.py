import pandas as pd
import matplotlib.dates as mdates

from lib.utils.visualization.constant import FIXED_Y_AXIS_LIMIT
from lib.constants import (
    DATE_LABEL,
    ESTIMATE_LABEL,
    ESTIMATE_PREDICTED_LABEL,
    Kalman,
)


def kalman_fit_plot(
    axs,
    data,
    options,
):
    data[DATE_LABEL] = pd.to_datetime(data[DATE_LABEL])

    axs.plot(
        data[DATE_LABEL],
        data[ESTIMATE_LABEL],
        label="Kalman Estimate",
        linestyle="-",
        color="blue",
    )

    axs.plot(
        data[DATE_LABEL],
        data[ESTIMATE_PREDICTED_LABEL],
        label="Kalman Predicted Estimate",
        linestyle="-",
        color="green",
    )

    axs.scatter(
        data[(data[Kalman.Z.value] != 0)][DATE_LABEL],
        data[(data[Kalman.Z.value] != 0)][Kalman.Z.value],
        label="Observed",
        s=13,
        color="red",
    )

    axs.xaxis.set_major_locator(mdates.AutoDateLocator())
    axs.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axs.tick_params(axis="x", labelsize=8)

    if options.get("fixed_y_axis", False):
        axs.set_ylim(0, options.get("fixed_y_axis_limit", FIXED_Y_AXIS_LIMIT))

    axs.set_title(options.get("title", ""))
