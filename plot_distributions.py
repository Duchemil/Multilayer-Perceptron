"""plot_distributions.py

Quick script to visualize the distribution of each column in a CSV using matplotlib.

Usage:
    python plot_distributions.py [path/to/data.csv]

Behavior:
 - If the CSV has no header (like your `data.csv`), the script will assign
   names: `id`, `diagnosis`, `feat_3`, `feat_4`, ...
 - Numeric columns are plotted as histograms.
 - Non-numeric / low-cardinality columns are plotted as bar charts of value counts.
 - The `id` column is skipped by default.
"""

import argparse
import math
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def assign_column_names(df: pd.DataFrame) -> List[str]:
    n = df.shape[1]
    if n >= 2:
        names = ["id", "diagnosis"] + [f"feat_{i}" for i in range(3, n + 1)]
    else:
        # fallback generic names
        names = [f"col_{i+1}" for i in range(n)]
    return names


def is_numeric_series(s: pd.Series) -> bool:
    # consider numeric if at least one value converts to numeric (not all NaN)
    converted = pd.to_numeric(s, errors="coerce")
    return converted.notna().any()


def plot_distributions(path: str, skip_id: bool = True, ncols: int = 3):
    df = pd.read_csv(path, header=None)

    # assign friendly column names if none present
    df.columns = assign_column_names(df)

    # optionally skip the id column
    cols = list(df.columns)
    if skip_id and "id" in cols:
        cols.remove("id")

    num_plots = len(cols)
    if num_plots == 0:
        raise SystemExit("No columns to plot.")

    ncols = max(1, int(ncols))
    nrows = math.ceil(num_plots / ncols)
    figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # flatten axes into a simple list of Axes objects (handle scalar, 1D or 2D ndarray)
    if isinstance(axes, plt.Axes):
        flat_axes = [axes]
    elif isinstance(axes, np.ndarray):
        flat_axes = axes.ravel().tolist()
    else:
        # fallback for lists/tuples
        flat_axes = [ax for row in axes for ax in (row if isinstance(row, (list, tuple)) else [row])]

    for ax, col in zip(flat_axes, cols):
        s = df[col]
        if is_numeric_series(s):
            s_numeric = pd.to_numeric(s, errors="coerce").dropna()
            if s_numeric.empty:
                ax.text(0.5, 0.5, "no numeric data", ha="center")
                ax.set_title(col)
                continue
            bins = 'auto'
            ax.hist(s_numeric, bins=bins, color="#4C72B0", alpha=0.8)
            ax.set_title(f"{col} (numeric)")
            ax.set_ylabel("count")
            # annotate mean / median
            mean = s_numeric.mean()
            median = s_numeric.median()
            ax.axvline(mean, color="red", linestyle="--", linewidth=1)
            ax.axvline(median, color="green", linestyle=":", linewidth=1)
            ax.legend([f"mean={mean:.3g}", f"med={median:.3g}"], loc="best")
        else:
            counts = s.astype(str).value_counts().sort_values(ascending=False)
            counts.plot.bar(ax=ax, color="#55A868", alpha=0.9)
            ax.set_title(f"{col} (categorical)")
            ax.set_ylabel("count")

    # hide any unused axes
    for ax in flat_axes[len(cols):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot distributions for each column in a CSV file.")
    parser.add_argument("csv", nargs="?", default="data.csv", help="Path to the CSV file (default: data.csv)")
    parser.add_argument("--no-skip-id", dest="skip_id", action="store_false", help="Do not skip the first id column")
    parser.add_argument("--cols", type=int, default=3, help="Number of subplot columns (default: 3)")
    args = parser.parse_args()

    path = args.csv
    if not os.path.isabs(path):
        # make relative to script location
        path = os.path.join(os.path.dirname(__file__), path)

    if not os.path.exists(path):
        raise SystemExit(f"CSV file not found: {path}")

    plot_distributions(path, skip_id=args.skip_id, ncols=args.cols)


if __name__ == "__main__":
    main()
