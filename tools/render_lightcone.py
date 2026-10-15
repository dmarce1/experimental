#!/usr/bin/env python3
# CosmicTiger - A cosmological N-Body code
# Copyright (C) 2026  Dominic C. Marcello
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Render one completed lightcone series as equal-solid-angle sky histograms."""

import argparse
import csv
import math
import os
from pathlib import Path
import sys

# Rendering is serial, including numerical-library helpers.
for variable in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np


def collectCatalogs(paths, longitudeBins):
    latitudeBins = longitudeBins // 2
    longitudeEdges = np.linspace(-math.pi, math.pi, longitudeBins + 1)
    sineEdges = np.linspace(-1.0, 1.0, latitudeBins + 1)
    counts = np.zeros((longitudeBins, latitudeBins))
    weights = np.zeros_like(counts)
    series = None
    files = 0
    particles = 0
    for path in sorted(set(Path(value).resolve() for value in paths)):
        suffix = ".crossings.csv"
        if not path.name.endswith(suffix):
            raise ValueError(f"Expected a crossing catalog: {path}")
        base = str(path)[:-len(suffix)]
        if not Path(base + ".complete").is_file():
            continue
        currentSeries = base.rsplit(".step", 1)[0]
        if series is not None and currentSeries != series:
            raise ValueError("Input mixes different runs; render one output series at a time")
        series = currentSeries
        with path.open() as stream:
            lines = (line for line in stream if line.strip() and not line.startswith("#"))
            rows = csv.DictReader(lines)
            for row in rows:
                position = [float(row[name]) for name in ("x", "y", "z")]
                weight = float(row["inverseRadiusSquared"])
                if not all(-sys.float_info.max <= value <= sys.float_info.max for value in position):
                    raise ValueError(f"Invalid position in {path}")
                if not 0.0 <= weight <= sys.float_info.max:
                    raise ValueError(f"Invalid weight in {path}")
                radius = math.hypot(*position)
                if not 0.0 < radius <= sys.float_info.max:
                    raise ValueError(f"Invalid crossing radius in {path}")
                longitude = math.atan2(position[1], position[0])
                sineLatitude = min(1.0, max(-1.0, position[2] / radius))
                x = min(longitudeBins - 1, int((longitude + math.pi) * longitudeBins / (2.0 * math.pi)))
                y = min(latitudeBins - 1, int((sineLatitude + 1.0) * latitudeBins / 2.0))
                counts[x, y] += 1.0
                with np.errstate(over="raise", invalid="raise"):
                    weights[x, y] += weight
                particles += 1
        files += 1
    if files == 0:
        raise ValueError("No completed crossing catalogs were found")
    return longitudeEdges, np.arcsin(sineEdges), counts, weights, particles, files


def render(paths, outputDirectory, longitudeBins=128, title="Lightcone"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patheffects import withStroke
    from matplotlib.colors import LogNorm, Normalize

    if longitudeBins < 4 or longitudeBins % 2:
        raise ValueError("Longitude bin count must be even and at least four")
    longitude, latitude, counts, weights, particles, files = collectCatalogs(paths, longitudeBins)
    outputDirectory = Path(outputDirectory)
    outputDirectory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, values, label in (
        ("counts", counts, "Particle crossings per angular bin"),
        ("weights", weights, "Sum of inverse squared radius per angular bin [box lengths⁻²]"),
    ):
        positive = values[values > 0]
        norm = LogNorm(vmin=positive.min(), vmax=max(positive.max(), positive.min() * 1.01)) if positive.size else Normalize(0, 1)
        figure = plt.figure(figsize=(12, 7), facecolor="white")
        axes = figure.add_subplot(111, projection="mollweide")
        axes.set_facecolor("#edf0f3")
        plot = axes.pcolormesh(longitude, latitude, np.ma.masked_less_equal(values.T, 0), shading="flat", cmap="magma", norm=norm)
        axes.grid(alpha=0.25)
        for labelText in axes.get_xticklabels():
            labelText.set_color("white")
            labelText.set_path_effects([withStroke(linewidth=2, foreground="black")])
        segmentLabel = "segment" if files == 1 else "segments"
        axes.set_title(f"{title} · {name}\n{particles:,} crossings · {files} completed {segmentLabel}", pad=24)
        axes.set_xlabel("Longitude")
        axes.set_ylabel("Latitude")
        figure.colorbar(plot, ax=axes, orientation="horizontal", pad=0.12, shrink=0.8, label=label)
        figure.text(0.5, 0.035, "Mollweide projection · equal-solid-angle longitude / sin(latitude) bins · empty bins are gray", ha="center", fontsize=9)
        filename = outputDirectory / f"lightcone_{name}.png"
        figure.savefig(filename, dpi=160, bbox_inches="tight")
        plt.close(figure)
        outputs.append(filename)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogs", nargs="+", help="Crossing CSV files from one run; incomplete segments are skipped")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--longitude-bins", type=int, default=128)
    parser.add_argument("--title", default="Lightcone")
    arguments = parser.parse_args()
    try:
        for filename in render(arguments.catalogs, arguments.output_directory, arguments.longitude_bins, arguments.title):
            print(filename)
    except (ValueError, OSError, KeyError, FloatingPointError) as error:
        parser.exit(1, f"Lightcone rendering failed: {error}\n")


if __name__ == "__main__":
    main()
