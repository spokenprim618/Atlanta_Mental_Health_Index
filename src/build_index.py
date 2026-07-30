"""
build_index.py

Computes a composite Mental Health Index (MHI) for the 10 Metro Atlanta
counties from the raw indicator dataset.

INPUT:
    data/metro_atlanta_mental_health_index_raw.csv
        Wide-format table: 1 row per county, plus a "Georgia (state average)"
        summary row, with columns for each indicator's value / data year / source.

METHOD:
    We only have clean, directly-comparable county-level numeric values for
    5 of the original 8 target indicators:
        - Poverty rate (%)                        [need]
        - Uninsured rate, under 65 (%)             [need]
        - Frequent mental distress (%)             [need]
        - Unemployment rate (%)                    [need]
        - Mental health providers per 100,000      [capacity]

    The other 3 (licensed facility counts, precise HPSA designation status,
    inpatient psych beds/100k) were not available as clean, distinct
    county-level numbers -- see README.md "Data Limitations" for why -- so
    they are excluded from the *numeric* index calculation, though they're
    still documented in the raw dataset as context/notes columns.

    Steps:
      1. Min-max normalize each of the 4 "need" indicators to a 0-1 scale
         across the 10 counties (higher = worse).
      2. Min-max normalize provider density (capacity) the same way, then
         invert it (1 - normalized) so higher = greater provider scarcity,
         matching the "higher = worse" direction of the other components.
      3. Average the 5 normalized components with equal weights (20% each).
      4. Rescale to a 0-100 score for readability. Higher score = greater
         mental health vulnerability / unmet need.

    This is intentionally a simple, transparent first-pass methodology.
    See README.md "Suggested Improvements" for ways to make the weighting
    more rigorous (PCA, sensitivity analysis, z-score standardization, etc).

OUTPUT:
    outputs/mental_health_index_final.csv
        County-level table with raw indicator values + MHI score + rank.
"""

import pandas as pd

RAW_DATA_PATH = "../data/metro_atlanta_mental_health_index_raw.csv"
OUTPUT_PATH = "../outputs/mental_health_index_final.csv"

# Indicators used in the index, and the direction of "worse"
NEED_COLUMNS = [
    "Poverty_Rate_Pct",
    "Uninsured_Rate_Under65_Pct",
    "Frequent_Mental_Distress_Pct",
    "Unemployment_Rate_Pct",
]
CAPACITY_COLUMN = "MH_Providers_per_100k"


def minmax_normalize(series: pd.Series) -> pd.Series:
    """Scale a series to the 0-1 range."""
    return (series - series.min()) / (series.max() - series.min())


def build_index(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the composite MHI score and rank for each county."""
    result = df[["County"] + NEED_COLUMNS + [CAPACITY_COLUMN]].copy()

    normalized_component_cols = []
    for col in NEED_COLUMNS:
        norm_col = f"{col}_norm"
        result[norm_col] = minmax_normalize(result[col])
        normalized_component_cols.append(norm_col)

    # Invert capacity so higher = greater need (provider scarcity)
    result["Provider_Scarcity_norm"] = 1 - minmax_normalize(result[CAPACITY_COLUMN])
    normalized_component_cols.append("Provider_Scarcity_norm")

    result["MHI_raw"] = result[normalized_component_cols].mean(axis=1)
    result["MHI_Score_0to100"] = (result["MHI_raw"] * 100).round(1)
    result["Rank"] = result["MHI_Score_0to100"].rank(ascending=False).astype(int)

    return result.drop(columns=normalized_component_cols + ["MHI_raw"])


def main():
    df = pd.read_csv(RAW_DATA_PATH)
    counties_only = df[df["County"] != "Georgia (state average)"].reset_index(drop=True)

    scored = build_index(counties_only)
    scored = scored.sort_values("Rank")

    scored.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")
    print(scored.to_string(index=False))


if __name__ == "__main__":
    main()
