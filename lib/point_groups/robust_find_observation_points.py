"""
Robust observation point finder that dynamically determines stable/unstable thresholds
and finds specified numbers of each type of point.
"""

import os
import shutil
import random
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import ee
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from enum import Enum
import sys

from lib.image_collections import COLLECTIONS
from lib.study_areas import PNW, RANDONIA

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)


class Stability(Enum):
    STABLE = "stable"
    UNSTABLE = "unstable"


@dataclass
class PointFinderConfig:
    """Configuration for robust point finding."""

    num_stable_points: int
    num_unstable_points: int
    study_area_name: str = "PNW"
    first_year: int = 2017
    last_year: int = 2018
    collection_name: str = "Randonia_l8_l9_2017_2018_swir"
    minimum_measurement_count: int = 15
    change_proxy_scale_meters: int = 30
    max_candidates_per_iteration: int = 500
    max_iterations: int = 20
    stable_percentile_threshold: float = 40.0  # Bottom 40% are stable
    unstable_percentile_threshold: float = 60.0  # Top 40% are unstable
    output_base_dir: str = "lib/point_groups/new"


class RobustPointFinder:
    """Finds stable and unstable observation points using dynamic thresholding."""

    def __init__(self, config: PointFinderConfig):
        self.config = config
        self.study_area = self._get_study_area()
        self.image_collection = COLLECTIONS[config.collection_name]
        self.output_dirs = self._setup_output_directories()

        # Results tracking
        self.found_stable_points = []
        self.found_unstable_points = []
        self.change_thresholds = {}

    def _get_study_area(self) -> List[List[float]]:
        """Get study area coordinates."""
        if self.config.study_area_name.upper() == "RANDONIA":
            return RANDONIA["coords"]
        elif self.config.study_area_name.upper() == "PNW":
            return PNW["coords"]
        else:
            raise ValueError(f"Unknown study area: {self.config.study_area_name}")

    def _setup_output_directories(self) -> Dict[str, str]:
        """Create output directories for stable and unstable points."""
        script_directory = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_directory, self.config.output_base_dir)

        stable_dir = os.path.join(base_dir, "stable")
        unstable_dir = os.path.join(base_dir, "unstable")

        os.makedirs(stable_dir, exist_ok=True)
        os.makedirs(unstable_dir, exist_ok=True)

        return {"stable": stable_dir, "unstable": unstable_dir}

    def _compute_change_magnitude_distribution(self) -> Tuple[float, float]:
        """Compute change magnitude distribution to determine dynamic thresholds."""
        study_area_geom = ee.Geometry.Polygon(self.study_area)

        # Define year boundaries
        first_year_start = ee.Date.fromYMD(self.config.first_year, 1, 1)
        first_year_end = ee.Date.fromYMD(self.config.first_year + 1, 1, 1)
        last_year_start = ee.Date.fromYMD(self.config.last_year, 1, 1)
        last_year_end = ee.Date.fromYMD(self.config.last_year + 1, 1, 1)

        # Compute annual means
        first_year_mean = (
            self.image_collection.filterDate(first_year_start, first_year_end)
            .select("swir")
            .mean()
        )
        last_year_mean = (
            self.image_collection.filterDate(last_year_start, last_year_end)
            .select("swir")
            .mean()
        )

        # Change magnitude and observation count
        change_magnitude = last_year_mean.subtract(first_year_mean).abs()
        count_all = self.image_collection.select("swir").reduce(ee.Reducer.count())
        sufficient_obs_mask = count_all.gte(self.config.minimum_measurement_count)

        # Sample change magnitudes where we have sufficient observations
        masked_change = change_magnitude.updateMask(sufficient_obs_mask)

        # Sample a large number of points to get distribution
        sample_size = 1000
        sampled = masked_change.sample(
            region=study_area_geom,
            scale=self.config.change_proxy_scale_meters,
            numPixels=sample_size,
            seed=42,
        )

        # Get the values and compute percentiles
        values = sampled.aggregate_array("swir").getInfo()

        if not values:
            raise RuntimeError(
                "No valid change magnitude values found. Check image collection and parameters."
            )

        stable_threshold = np.percentile(
            values, self.config.stable_percentile_threshold
        )
        unstable_threshold = np.percentile(
            values, self.config.unstable_percentile_threshold
        )

        print(f"Change magnitude distribution:")
        print(
            f"  Stable threshold (p{self.config.stable_percentile_threshold}): {stable_threshold:.4f}"
        )
        print(
            f"  Unstable threshold (p{self.config.unstable_percentile_threshold}): {unstable_threshold:.4f}"
        )
        print(
            f"  Min: {np.min(values):.4f}, Max: {np.max(values):.4f}, Median: {np.median(values):.4f}"
        )
        print(f"  Sample size: {len(values)} points")
        print(
            f"  Values above unstable threshold: {np.sum(np.array(values) >= unstable_threshold)}"
        )
        print(
            f"  Values below stable threshold: {np.sum(np.array(values) < stable_threshold)}"
        )

        return stable_threshold, unstable_threshold

    def _get_candidate_points(
        self, stability_type: Stability, threshold: float
    ) -> ee.FeatureCollection:
        """Get candidate points for a specific stability type."""
        study_area_geom = ee.Geometry.Polygon(self.study_area)

        # Define year boundaries
        first_year_start = ee.Date.fromYMD(self.config.first_year, 1, 1)
        first_year_end = ee.Date.fromYMD(self.config.first_year + 1, 1, 1)
        last_year_start = ee.Date.fromYMD(self.config.last_year, 1, 1)
        last_year_end = ee.Date.fromYMD(self.config.last_year + 1, 1, 1)

        # Compute annual means
        first_year_mean = (
            self.image_collection.filterDate(first_year_start, first_year_end)
            .select("swir")
            .mean()
        )
        last_year_mean = (
            self.image_collection.filterDate(last_year_start, last_year_end)
            .select("swir")
            .mean()
        )

        # Change magnitude and observation count
        change_magnitude = last_year_mean.subtract(first_year_mean).abs()
        count_all = self.image_collection.select("swir").reduce(ee.Reducer.count())
        sufficient_obs_mask = count_all.gte(self.config.minimum_measurement_count)

        # Apply threshold based on stability type
        if stability_type == Stability.STABLE:
            candidate_mask = change_magnitude.lt(threshold)
        else:  # UNSTABLE
            candidate_mask = change_magnitude.gte(threshold)

        # Combine masks
        masked = change_magnitude.updateMask(candidate_mask).updateMask(
            sufficient_obs_mask
        )

        # Sample candidate points
        sampled = masked.sample(
            region=study_area_geom,
            scale=self.config.change_proxy_scale_meters,
            numPixels=self.config.max_candidates_per_iteration,
            geometries=True,
            seed=random.randint(0, 1_000_000),
        )

        result_fc = ee.FeatureCollection(sampled)

        # Debug: print how many candidate points were found
        candidate_count = result_fc.size().getInfo()
        print(
            f"    Found {candidate_count} candidate {stability_type.value} points (threshold: {threshold:.4f})"
        )

        return result_fc

    def _sample_time_series_for_points(
        self, geometries_fc: ee.FeatureCollection
    ) -> List[Dict]:
        """Sample time series for given geometries."""

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

        measurements_fc = ee.FeatureCollection(
            self.image_collection.map(process_image).flatten()
        )
        features = measurements_fc.getInfo()["features"]

        # Convert to structured data
        measurements = []
        for feature in features:
            if (
                "swir" in feature["properties"]
                and feature["properties"].get("millis") is not None
            ):
                measurements.append(
                    {
                        "longitude": feature["geometry"]["coordinates"][0],
                        "latitude": feature["geometry"]["coordinates"][1],
                        "date": feature["properties"]["millis"],
                        "swir": feature["properties"]["swir"],
                    }
                )

        return measurements

    def _validate_point_stability(
        self, data: pd.DataFrame, stability_type: Stability
    ) -> bool:
        """Validate that a point meets stability criteria based on time series analysis."""
        if len(data) < self.config.minimum_measurement_count:
            return False

        data["year"] = pd.to_datetime(data["date"], unit="ms").dt.year
        mean_swir_first_year = data[data["year"] == self.config.first_year][
            "swir"
        ].mean()
        mean_swir_last_year = data[data["year"] == self.config.last_year]["swir"].mean()

        difference_in_mean_swir = abs(mean_swir_first_year - mean_swir_last_year)

        # Use the same thresholds we computed from the distribution
        if stability_type == Stability.STABLE:
            is_stable = difference_in_mean_swir < self.change_thresholds["stable"]
            if not is_stable:
                print(
                    f"      Point rejected: change={difference_in_mean_swir:.4f} >= stable_thresh={self.change_thresholds['stable']:.4f}"
                )
            return is_stable
        else:  # UNSTABLE
            is_unstable = difference_in_mean_swir >= self.change_thresholds["unstable"]
            if not is_unstable:
                print(
                    f"      Point rejected: change={difference_in_mean_swir:.4f} < unstable_thresh={self.change_thresholds['unstable']:.4f}"
                )
            return is_unstable

    def _plot_point_timeseries(self, data: pd.DataFrame, stability_type: Stability):
        """Generate and save time series plots for a point."""
        fig, axs = plt.subplots(1, 2, figsize=(24, 8))
        data["date"] = pd.to_datetime(data["date"], unit="ms")

        color = "green" if stability_type == Stability.STABLE else "red"
        label = f"SWIR ({stability_type.value})"

        # Plot 1: Full range
        axs[0].scatter(data["date"], data["swir"], label=label, color=color)
        axs[0].xaxis.set_major_locator(mdates.AutoDateLocator())
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axs[0].tick_params(axis="x", labelsize=8)
        axs[0].set_title(
            f"{data['longitude'].iloc[0]:.6f},{data['latitude'].iloc[0]:.6f}"
        )
        axs[0].legend()

        # Plot 2: Constrained range
        axs[1].scatter(data["date"], data["swir"], label=label, color=color)
        axs[1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axs[1].tick_params(axis="x", labelsize=8)
        axs[1].set_ylim(0, 0.5)
        axs[1].set_title(
            f"{data['longitude'].iloc[0]:.6f},{data['latitude'].iloc[0]:.6f}"
        )
        axs[1].legend()

        title = f"({data['longitude'].iloc[0]:.6f},{data['latitude'].iloc[0]:.6f})"
        output_dir = self.output_dirs[stability_type.value]

        print(f"Found {stability_type.value} point: {title}")
        fig.savefig(f"{output_dir}/{title}.png")
        plt.close(fig)

        return title

    def _find_points_of_type(
        self, stability_type: Stability, target_count: int
    ) -> List[str]:
        """Find the specified number of points of a given stability type."""
        found_points = []
        iteration = 0

        print(f"\nFinding {target_count} {stability_type.value} points...")

        while (
            len(found_points) < target_count and iteration < self.config.max_iterations
        ):
            iteration += 1
            print(
                f"  Iteration {iteration}: Looking for {stability_type.value} points..."
            )

            # Get candidate points
            threshold = self.change_thresholds[stability_type.value]
            candidate_points = self._get_candidate_points(stability_type, threshold)

            if candidate_points.size().getInfo() == 0:
                print(
                    f"    No candidate points found for {stability_type.value} in iteration {iteration}"
                )
                continue

            # Sample time series for candidates
            measurements = self._sample_time_series_for_points(candidate_points)

            if not measurements:
                print(f"    No measurements found in iteration {iteration}")
                continue

            # Group by location and validate
            df = pd.DataFrame(measurements)
            grouped_measurements = df.groupby(["longitude", "latitude"])

            for point, data in grouped_measurements:
                if len(found_points) >= target_count:
                    break

                if self._validate_point_stability(data, stability_type):
                    point_title = self._plot_point_timeseries(data, stability_type)
                    found_points.append(point_title)
                    print(
                        f"    Found {len(found_points)}/{target_count} {stability_type.value} points"
                    )

        if len(found_points) < target_count:
            print(
                f"  Warning: Only found {len(found_points)} out of {target_count} requested {stability_type.value} points"
            )

        return found_points

    def find_points(self) -> Dict[str, List[str]]:
        """Main method to find both stable and unstable points."""
        print("Computing dynamic thresholds from change magnitude distribution...")

        # Compute dynamic thresholds
        stable_thresh, unstable_thresh = self._compute_change_magnitude_distribution()
        self.change_thresholds = {"stable": stable_thresh, "unstable": unstable_thresh}

        # Find stable points
        if self.config.num_stable_points > 0:
            self.found_stable_points = self._find_points_of_type(
                Stability.STABLE, self.config.num_stable_points
            )

        # Find unstable points
        if self.config.num_unstable_points > 0:
            self.found_unstable_points = self._find_points_of_type(
                Stability.UNSTABLE, self.config.num_unstable_points
            )

        # Summary
        print(f"\n=== RESULTS ===")
        print(f"Found {len(self.found_stable_points)} stable points")
        print(f"Found {len(self.found_unstable_points)} unstable points")
        print(f"Stable threshold: {self.change_thresholds['stable']:.4f}")
        print(f"Unstable threshold: {self.change_thresholds['unstable']:.4f}")

        return {
            "stable": self.found_stable_points,
            "unstable": self.found_unstable_points,
            "thresholds": self.change_thresholds,
        }


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Find stable and unstable observation points"
    )
    parser.add_argument(
        "--stable", type=int, default=10, help="Number of stable points to find"
    )
    parser.add_argument(
        "--unstable", type=int, default=10, help="Number of unstable points to find"
    )
    parser.add_argument(
        "--study-area",
        type=str,
        default="RANDONIA",
        choices=["RANDONIA", "PNW"],
        help="Study area to use",
    )
    parser.add_argument(
        "--first-year", type=int, default=2017, help="First year for comparison"
    )
    parser.add_argument(
        "--last-year", type=int, default=2018, help="Last year for comparison"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="Randonia_l8_l9_2017_2018_swir",
        help="Image collection to use",
    )
    parser.add_argument(
        "--min-obs", type=int, default=15, help="Minimum observations required"
    )
    parser.add_argument(
        "--stable-percentile",
        type=float,
        default=25.0,
        help="Percentile threshold for stable points",
    )
    parser.add_argument(
        "--unstable-percentile",
        type=float,
        default=75.0,
        help="Percentile threshold for unstable points",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=20, help="Maximum iterations to attempt"
    )

    args = parser.parse_args()

    # Create configuration
    config = PointFinderConfig(
        num_stable_points=args.stable,
        num_unstable_points=args.unstable,
        study_area_name=args.study_area,
        first_year=args.first_year,
        last_year=args.last_year,
        collection_name=args.collection,
        minimum_measurement_count=args.min_obs,
        stable_percentile_threshold=args.stable_percentile,
        unstable_percentile_threshold=args.unstable_percentile,
        max_iterations=args.max_iterations,
    )

    # Find points
    finder = RobustPointFinder(config)
    results = finder.find_points()

    print("\nPoint finding completed!")


if __name__ == "__main__":
    main()
