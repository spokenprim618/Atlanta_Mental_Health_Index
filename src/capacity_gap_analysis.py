"""
capacity_gap_analysis.py

Creates supplemental analyses using mental health resource indicators that
were not included directly in the Mental Health Index (MHI) calculation.

INPUT:
    outputs/mental_health_index_final.csv

OUTPUT:
    outputs/capacity_profile.csv
    outputs/priority_gap_rankings.csv
    outputs/facility_concentration.csv
    outputs/mhpsa_summary.csv
    outputs/psych_bed_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FINAL_PATH = PROJECT_ROOT / "outputs" / "mental_health_index_final.csv"
RAW_PATH = PROJECT_ROOT / "data" / "metro_atlanta_mental_health_index_raw.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

def load_analysis_dataframe():
    """
    Merge the final MHI scores/ranks with all contextual variables from the
    raw dataset.
    """

    final_df = pd.read_csv(FINAL_PATH)
    raw_df = pd.read_csv(RAW_PATH)

    raw_df = raw_df[
        raw_df["County"] != "Georgia (state average)"
    ].copy()

    merged = pd.merge(
        final_df[
            [
                "County",
                "MHI_Score_0to100",
                "Rank",
            ]
        ],
        raw_df,
        on="County",
        how="left",
    )

    return merged

def create_capacity_profile(df: pd.DataFrame) -> pd.DataFrame:
    """County-level capacity profile table."""

    cols = [
        "County",
        "MH_Providers_per_100k",
        "Licensed_MH_Facilities_Count",
        "Psych_Beds_per_100k",
        "MHPSA_Designation_Status",
    ]

    profile = df[cols].copy()

    profile.to_csv(
        OUTPUT_DIR / "capacity_profile.csv",
        index=False,
    )

    return profile


def create_priority_gap_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank counties where mental health need most exceeds available capacity.
    """

    temp = df.copy()

    temp["Need_Score"] = temp[
        [
            "Poverty_Rate_Pct",
            "Uninsured_Rate_Under65_Pct",
            "Frequent_Mental_Distress_Pct",
            "Unemployment_Rate_Pct",
        ]
    ].mean(axis=1)

    need_z = (
        temp["Need_Score"] - temp["Need_Score"].mean()
    ) / temp["Need_Score"].std()

    provider_z = (
        temp["MH_Providers_per_100k"]
        - temp["MH_Providers_per_100k"].mean()
    ) / temp["MH_Providers_per_100k"].std()

    temp["Priority_Gap"] = need_z - provider_z

    rankings = temp[
        [
            "County",
            "Need_Score",
            "MH_Providers_per_100k",
            "MHI_Score_0to100",
            "Priority_Gap",
        ]
    ].sort_values(
        "Priority_Gap",
        ascending=False,
    )

    rankings.insert(0, "Priority_Rank", range(1, len(rankings) + 1))

    rankings.to_csv(
        OUTPUT_DIR / "priority_gap_rankings.csv",
        index=False,
    )

    return rankings


def create_facility_concentration(df):
    """Share of all metro facilities located in each county."""

    temp = df.copy()

    facilities = pd.to_numeric(
        temp["Licensed_MH_Facilities_Count"],
        errors="coerce",
    )

    if facilities.notna().sum() == 0:
        print(
            "Skipping facility concentration analysis: "
            "no numeric facility counts available."
        )
        return None

    temp["Licensed_MH_Facilities_Count"] = facilities

    total_facilities = facilities.sum()

    temp["Facility_Share_Pct"] = (
        facilities / total_facilities * 100
    )

    concentration = temp[
        [
            "County",
            "Licensed_MH_Facilities_Count",
            "Facility_Share_Pct",
        ]
    ].sort_values(
        "Facility_Share_Pct",
        ascending=False,
    )

    concentration.to_csv(
        OUTPUT_DIR / "facility_concentration.csv",
        index=False,
    )

    return concentration

def create_mhpsa_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of counties by MHPSA designation."""

    summary = (
        df["MHPSA_Designation_Status"]
        .fillna("Unknown")
        .value_counts(dropna=False)
        .reset_index()
    )

    summary.columns = [
        "MHPSA_Designation_Status",
        "County_Count",
    ]

    summary.to_csv(
        OUTPUT_DIR / "mhpsa_summary.csv",
        index=False,
    )

    return summary


def create_psych_bed_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics and correlations for psychiatric bed availability.
    """

    beds = pd.to_numeric(
        df["Psych_Beds_per_100k"],
        errors="coerce",
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "Mean",
                "Median",
                "Minimum",
                "Maximum",
                "Std_Dev",
                "Correlation_With_MHI",
                "Correlation_With_Frequent_Mental_Distress",
            ],
            "Value": [
                beds.mean(),
                beds.median(),
                beds.min(),
                beds.max(),
                beds.std(),
                beds.corr(df["MHI_Score_0to100"]),
                beds.corr(df["Frequent_Mental_Distress_Pct"]),
            ],
        }
    )

    summary["Value"] = summary["Value"].round(3)

    summary.to_csv(
        OUTPUT_DIR / "psych_bed_summary.csv",
        index=False,
    )

    return summary


def main():
    df = load_analysis_dataframe()

    create_capacity_profile(df)
    create_priority_gap_rankings(df)
    create_facility_concentration(df)
    create_mhpsa_summary(df)
    create_psych_bed_summary(df)

    print("\nGenerated:")
    print(" - capacity_profile.csv")
    print(" - priority_gap_rankings.csv")
    print(" - facility_concentration.csv")
    print(" - mhpsa_summary.csv")
    print(" - psych_bed_summary.csv")


if __name__ == "__main__":
    main()