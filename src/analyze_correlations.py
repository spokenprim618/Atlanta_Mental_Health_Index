"""
analyze_correlations.py

Quick exploratory check of how the 5 index-input indicators relate to each
other across the 10 counties. Useful for spotting redundancy between
indicators before finalizing index weights (see README.md "Suggested
Improvements" re: PCA-based weighting).

INPUT:
    data/metro_atlanta_mental_health_index_raw.csv

OUTPUT:
    Prints a Pearson correlation matrix to stdout.
"""

import pandas as pd

RAW_DATA_PATH = "data/metro_atlanta_mental_health_index_raw.csv"

INDICATOR_COLUMNS = [
    "Poverty_Rate_Pct",
    "Uninsured_Rate_Under65_Pct",
    "Frequent_Mental_Distress_Pct",
    "Unemployment_Rate_Pct",
    "MH_Providers_per_100k",
]


def main():
    df = pd.read_csv(RAW_DATA_PATH)
    counties_only = df[df["County"] != "Georgia (state average)"]

    corr = counties_only[INDICATOR_COLUMNS].corr(method="pearson").round(2)
    print("Pearson correlation matrix (n=10 counties):\n")
    print(corr)


if __name__ == "__main__":
    main()
