"""
analyze_correlations.py

Quick exploratory check of how the 5 index-input indicators relate to each
other across the 10 counties. Useful for spotting redundancy between
indicators before finalizing index weights (see README.md "Suggested
Improvements" re: PCA-based weighting).

INPUT:
    data/metro_atlanta_mental_health_index_raw.csv

OUTPUT:
    - Prints a Pearson correlation matrix to stdout.
    - Generates a correlation heatmap.
    - Saves the heatmap to ../outputs/correlation_heatmap.png
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAW_DATA_PATH = "../data/metro_atlanta_mental_health_index_raw.csv"
OUTPUT_DIR = "../outputs"
OUTPUT_FIGURE = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")

INDICATOR_COLUMNS = [
    "Poverty_Rate_Pct",
    "Uninsured_Rate_Under65_Pct",
    "Frequent_Mental_Distress_Pct",
    "Unemployment_Rate_Pct",
    "MH_Providers_per_100k",
]


def main():
    # Load data
    df = pd.read_csv(RAW_DATA_PATH)

    # Remove statewide average row so correlations are based only on counties
    counties_only = df[df["County"] != "Georgia (state average)"].copy()

    # Calculate Pearson correlations
    corr = counties_only[INDICATOR_COLUMNS].corr(method="pearson").round(2)

    print("Pearson correlation matrix (n=10 counties):\n")
    print(corr)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        fmt=".2f",
    )

    plt.title("Pearson Correlation Heatmap of Mental Health Index Indicators")
    plt.tight_layout()

    # Save figure
    plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nHeatmap saved to: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()