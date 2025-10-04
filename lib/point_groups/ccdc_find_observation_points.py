"""
CCDC-based observation point finder that uses change segments to classify points
as stable or unstable based on whether they experienced change during a specified interval.
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
from datetime import datetime

from lib.image_collections import COLLECTIONS
from lib.study_areas import PNW, RANDONIA
from lib.utils.ee.ccdc_utils import (
    build_ccd_image,
    get_multi_coefs,
    build_segment_tag,
    get_segments_for_coordinates,
)
from lib.utils.ee.dates import convert_date
from lib.constants import CCDC, HARMONIC_TAGS

ee.Initialize(opt_url=ee.data.HIGH_VOLUME_API_BASE_URL)


class Stability(Enum):
    STABLE = "stable"
    UNSTABLE = "unstable"


@dataclass
class CCDCConfig:
    """Configuration for CCDC-based point finding."""

    num_stable_points: int
    num_unstable_points: int
    study_area_name: str = "PNW"
    start_year: int = 2017
    end_year: int = 2018
    ccdc_collection_name: str = "CCDC_Global"
    image_collection_name: str = "PNW_L8_L9_2017_2018"
    minimum_measurement_count: int = 15
    change_proxy_scale_meters: int = 100
    max_candidates_per_iteration: int = 200
    max_iterations: int = 20
    output_base_dir: str = "lib/point_groups/ccdc_results"


class CCDCChangeDetector:
    """Detects change using CCDC segments for a given time interval."""

    def __init__(self, config: CCDCConfig):
        self.config = config
        self.study_area = self._get_study_area()
        self.ccdc_collection = COLLECTIONS[config.ccdc_collection_name]
        self.image_collection = COLLECTIONS[config.image_collection_name]
        self.output_dirs = self._setup_output_directories()

        # Convert years to milliseconds (as Python values)
        self.start_date_ms = ee.Date.fromYMD(config.start_year, 1, 1).millis().getInfo()
        self.end_date_ms = ee.Date.fromYMD(config.end_year + 1, 1, 1).millis().getInfo()

        # Results tracking
        self.found_stable_points = []
        self.found_unstable_points = []

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

    def _build_ccdc_image(self) -> ee.Image:
        """Build CCDC image with segment information."""
        bands = ["SWIR1"]
        segments_count = 10
        segments = build_segment_tag(segments_count)

        # Get CCDC mosaic and clip to study area to reduce memory usage
        study_area_geom = ee.Geometry.Polygon(self.study_area)
        ccdc_mosaic = self.ccdc_collection.mosaic().clip(study_area_geom)
        ccdc_image = build_ccd_image(ccdc_mosaic, segments_count, bands)

        return ccdc_image

    def _detect_change_in_interval(self, ccdc_image: ee.Image) -> ee.Image:
        """Create a mask indicating where change occurred during the specified interval."""
        bands = ["SWIR1"]
        segments_count = 10
        segments = build_segment_tag(segments_count)

        # Get segment start and end times
        start_bands = ccdc_image.select(".*_tStart").rename(segments)
        end_bands = ccdc_image.select(".*_tEnd").rename(segments)

        # Check if any segment has a start or end date within our interval
        # A change occurred if a segment starts or ends during our time period
        start_in_interval = start_bands.gte(self.start_date_ms).And(
            start_bands.lt(self.end_date_ms)
        )
        end_in_interval = end_bands.gte(self.start_date_ms).And(
            end_bands.lt(self.end_date_ms)
        )

        # Any segment that starts or ends in interval indicates change
        change_mask = start_in_interval.Or(end_in_interval)

        # Reduce across segments - if any segment changed, pixel is unstable
        unstable_pixels = change_mask.reduce(ee.Reducer.anyNonZero())
        stable_pixels = change_mask.reduce(ee.Reducer.allNonZero()).Not()

        return unstable_pixels.rename("change_detected")

    def _get_candidate_points(self, stability_type: Stability) -> ee.FeatureCollection:
        """Get candidate points based on CCDC change detection."""
        study_area_geom = ee.Geometry.Polygon(self.study_area)

        # Build CCDC image and detect changes
        ccdc_image = self._build_ccdc_image()
        change_mask = self._detect_change_in_interval(ccdc_image)

        # Apply stability mask
        if stability_type == Stability.STABLE:
            candidate_mask = change_mask.Not()  # No change = stable
        else:  # UNSTABLE
            candidate_mask = change_mask  # Change detected = unstable

        # Sample candidate points with reduced memory footprint
        sampled = candidate_mask.sample(
            region=study_area_geom,
            scale=self.config.change_proxy_scale_meters,
            numPixels=self.config.max_candidates_per_iteration,
            geometries=True,
            seed=random.randint(0, 1_000_000),
            tileScale=4,  # Reduce tile scale to use less memory
        )

        result_fc = ee.FeatureCollection(sampled)

        # Debug: print how many candidate points were found
        candidate_count = result_fc.size().getInfo()
        print(
            f"    Found {candidate_count} candidate {stability_type.value} points (CCDC-based)"
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
                collection=geometries_fc, scale=10, geometries=True, tileScale=4
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

    def _validate_point_with_ccdc(
        self,
        data: pd.DataFrame,
        stability_type: Stability,
        point_coords: Tuple[float, float],
    ) -> bool:
        """Validate point using CCDC segment analysis."""
        if len(data) < self.config.minimum_measurement_count:
            return False

        # Get CCDC segments for this specific point
        try:
            segments = get_segments_for_coordinates(point_coords)
            if not segments:
                print(f"      No CCDC segments found for point {point_coords}")
                return False

            # Check if any segment starts or ends during our interval
            change_detected = False
            for start_date, end_date in segments:
                # Convert to milliseconds for comparison
                start_ms = start_date * 1000 if start_date > 1e10 else start_date
                end_ms = end_date * 1000 if end_date > 1e10 else end_date

                # Check if segment change occurred during our interval
                if (
                    self.start_date_ms <= start_ms < self.end_date_ms
                    or self.start_date_ms <= end_ms < self.end_date_ms
                ):
                    change_detected = True
                    break

            # Validate against expected stability
            if stability_type == Stability.STABLE:
                is_valid = not change_detected
                if not is_valid:
                    print(
                        f"      Point rejected: CCDC detected change (expected stable)"
                    )
                return is_valid
            else:  # UNSTABLE
                is_valid = change_detected
                if not is_valid:
                    print(
                        f"      Point rejected: No CCDC change detected (expected unstable)"
                    )
                return is_valid

        except Exception as e:
            print(f"      Error validating point with CCDC: {e}")
            return False

    def _plot_point_timeseries_with_ccdc(
        self,
        data: pd.DataFrame,
        stability_type: Stability,
        point_coords: Tuple[float, float],
    ):
        """Generate and save time series plots with CCDC segment information."""
        fig, axs = plt.subplots(2, 2, figsize=(24, 16))
        data["date"] = pd.to_datetime(data["date"], unit="ms")

        color = "green" if stability_type == Stability.STABLE else "red"
        label = f"SWIR ({stability_type.value})"

        # Plot 1: Full range time series
        axs[0, 0].scatter(data["date"], data["swir"], label=label, color=color)
        axs[0, 0].xaxis.set_major_locator(mdates.AutoDateLocator())
        axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axs[0, 0].tick_params(axis="x", labelsize=8)
        axs[0, 0].set_title(f"Time Series: {point_coords[0]:.6f},{point_coords[1]:.6f}")
        axs[0, 0].legend()

        # Add vertical lines for our analysis interval
        axs[0, 0].axvline(
            pd.to_datetime(self.config.start_year, format="%Y"),
            color="blue",
            linestyle="--",
            alpha=0.7,
            label="Analysis Start",
        )
        axs[0, 0].axvline(
            pd.to_datetime(self.config.end_year, format="%Y"),
            color="blue",
            linestyle="--",
            alpha=0.7,
            label="Analysis End",
        )

        # Plot 2: Constrained range
        axs[0, 1].scatter(data["date"], data["swir"], label=label, color=color)
        axs[0, 1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        axs[0, 1].tick_params(axis="x", labelsize=8)
        axs[0, 1].set_ylim(0, 0.5)
        axs[0, 1].set_title("Constrained Range")
        axs[0, 1].legend()
        axs[0, 1].axvline(
            pd.to_datetime(self.config.start_year, format="%Y"),
            color="blue",
            linestyle="--",
            alpha=0.7,
        )
        axs[0, 1].axvline(
            pd.to_datetime(self.config.end_year, format="%Y"),
            color="blue",
            linestyle="--",
            alpha=0.7,
        )

        # Plot 3: CCDC segment timeline
        try:
            segments = get_segments_for_coordinates(point_coords)
            if segments:
                axs[1, 0].set_title("CCDC Segments")
                for i, (start_date, end_date) in enumerate(segments):
                    start_ms = start_date * 1000 if start_date > 1e10 else start_date
                    end_ms = end_date * 1000 if end_date > 1e10 else end_date
                    start_dt = pd.to_datetime(start_ms, unit="ms")
                    end_dt = pd.to_datetime(end_ms, unit="ms")

                    # Check if segment overlaps with our analysis period
                    analysis_start = pd.to_datetime(self.config.start_year, format="%Y")
                    analysis_end = pd.to_datetime(self.config.end_year, format="%Y")

                    if start_dt <= analysis_end and end_dt >= analysis_start:
                        # Segment overlaps with analysis period
                        axs[1, 0].barh(
                            i,
                            (end_dt - start_dt).days,
                            left=start_dt,
                            height=0.8,
                            color="red",
                            alpha=0.7,
                            label="Change Segment" if i == 0 else "",
                        )
                    else:
                        axs[1, 0].barh(
                            i,
                            (end_dt - start_dt).days,
                            left=start_dt,
                            height=0.8,
                            color="gray",
                            alpha=0.5,
                        )

                axs[1, 0].axvline(
                    analysis_start,
                    color="blue",
                    linestyle="--",
                    alpha=0.7,
                    label="Analysis Period",
                )
                axs[1, 0].axvline(analysis_end, color="blue", linestyle="--", alpha=0.7)
                axs[1, 0].legend()
                axs[1, 0].set_xlabel("Date")
                axs[1, 0].set_ylabel("Segment")
            else:
                axs[1, 0].text(
                    0.5,
                    0.5,
                    "No CCDC segments found",
                    ha="center",
                    va="center",
                    transform=axs[1, 0].transAxes,
                )
                axs[1, 0].set_title("CCDC Segments")
        except Exception as e:
            axs[1, 0].text(
                0.5,
                0.5,
                f"Error loading CCDC: {str(e)[:50]}...",
                ha="center",
                va="center",
                transform=axs[1, 0].transAxes,
            )
            axs[1, 0].set_title("CCDC Segments")

        # Plot 4: Summary info
        axs[1, 1].text(
            0.1,
            0.9,
            f"Point: {point_coords[0]:.6f}, {point_coords[1]:.6f}",
            transform=axs[1, 1].transAxes,
            fontsize=12,
        )
        axs[1, 1].text(
            0.1,
            0.8,
            f"Classification: {stability_type.value.upper()}",
            transform=axs[1, 1].transAxes,
            fontsize=12,
        )
        axs[1, 1].text(
            0.1,
            0.7,
            f"Analysis Period: {self.config.start_year}-{self.config.end_year}",
            transform=axs[1, 1].transAxes,
            fontsize=12,
        )
        axs[1, 1].text(
            0.1,
            0.6,
            f"Observations: {len(data)}",
            transform=axs[1, 1].transAxes,
            fontsize=12,
        )
        axs[1, 1].text(
            0.1,
            0.5,
            f"Data Range: {data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}",
            transform=axs[1, 1].transAxes,
            fontsize=10,
        )
        axs[1, 1].set_xlim(0, 1)
        axs[1, 1].set_ylim(0, 1)
        axs[1, 1].axis("off")

        title = f"({point_coords[0]:.6f},{point_coords[1]:.6f})"
        output_dir = self.output_dirs[stability_type.value]

        print(f"Found {stability_type.value} point: {title}")
        fig.savefig(f"{output_dir}/{title}_ccdc.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        return title

    def _find_points_of_type(
        self, stability_type: Stability, target_count: int
    ) -> List[str]:
        """Find the specified number of points of a given stability type."""
        found_points = []
        iteration = 0

        print(f"\nFinding {target_count} {stability_type.value} points using CCDC...")

        while (
            len(found_points) < target_count and iteration < self.config.max_iterations
        ):
            iteration += 1
            print(
                f"  Iteration {iteration}: Looking for {stability_type.value} points..."
            )

            # Get candidate points
            candidate_points = self._get_candidate_points(stability_type)

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

                point_coords = (point[0], point[1])
                if self._validate_point_with_ccdc(data, stability_type, point_coords):
                    point_title = self._plot_point_timeseries_with_ccdc(
                        data, stability_type, point_coords
                    )
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
        """Main method to find both stable and unstable points using CCDC."""
        print(
            f"Using CCDC to find stable/unstable points for {self.config.start_year}-{self.config.end_year}"
        )
        print(
            f"Analysis interval: {datetime.fromtimestamp(self.start_date_ms/1000).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(self.end_date_ms/1000).strftime('%Y-%m-%d')}"
        )

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
        print(f"\n=== CCDC RESULTS ===")
        print(f"Found {len(self.found_stable_points)} stable points")
        print(f"Found {len(self.found_unstable_points)} unstable points")
        print(f"Analysis period: {self.config.start_year}-{self.config.end_year}")
        print(f"Stable points: No CCDC change segments during analysis period")
        print(f"Unstable points: CCDC change segments detected during analysis period")

        return {
            "stable": self.found_stable_points,
            "unstable": self.found_unstable_points,
            "analysis_period": f"{self.config.start_year}-{self.config.end_year}",
            "method": "CCDC_segments",
        }


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Find stable and unstable observation points using CCDC segments"
    )
    parser.add_argument(
        "--stable", type=int, default=0, help="Number of stable points to find"
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
        "--start-year", type=int, default=2017, help="Start year for change detection"
    )
    parser.add_argument(
        "--end-year", type=int, default=2018, help="End year for change detection"
    )
    parser.add_argument(
        "--ccdc-collection",
        type=str,
        default="CCDC_Global",
        help="CCDC collection to use",
    )
    parser.add_argument(
        "--image-collection",
        type=str,
        default="Randonia_l8_l9_2017_2018_swir",
        help="Image collection for time series",
    )
    parser.add_argument(
        "--min-obs", type=int, default=50, help="Minimum observations required"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=20, help="Maximum iterations to attempt"
    )

    args = parser.parse_args()

    # Create configuration
    config = CCDCConfig(
        num_stable_points=args.stable,
        num_unstable_points=args.unstable,
        study_area_name=args.study_area,
        start_year=args.start_year,
        end_year=args.end_year,
        ccdc_collection_name=args.ccdc_collection,
        image_collection_name=args.image_collection,
        minimum_measurement_count=args.min_obs,
        max_iterations=args.max_iterations,
    )

    # Find points
    detector = CCDCChangeDetector(config)
    results = detector.find_points()

    print("\nCCDC-based point finding completed!")


if __name__ == "__main__":
    main()
