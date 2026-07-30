"""
visualize_capacity_gaps.py

Creates visualizations and table graphics from the supplemental
capacity-gap analysis outputs.

INPUT:
    outputs/capacity_profile.csv
    outputs/priority_gap_rankings.csv
    outputs/facility_concentration.csv
    outputs/mhpsa_summary.csv
    outputs/psych_bed_summary.csv

OUTPUT:
    outputs/fig_priority_gap_rankings.png
    outputs/fig_capacity_profile_heatmap.png
    outputs/fig_facility_concentration.png
    outputs/fig_mhpsa_designations.png

    outputs/table_capacity_profile.png
    outputs/table_priority_gap_top5.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CAPACITY_PROFILE = OUTPUT_DIR / "capacity_profile.csv"
PRIORITY_GAPS = OUTPUT_DIR / "priority_gap_rankings.csv"
FACILITY_CONCENTRATION = OUTPUT_DIR / "facility_concentration.csv"
MHPSA_SUMMARY = OUTPUT_DIR / "mhpsa_summary.csv"
PSYCH_BED_SUMMARY = OUTPUT_DIR / "psych_bed_summary.csv"

sns.set_theme(style="whitegrid")


def create_priority_gap_chart():
    df = pd.read_csv(PRIORITY_GAPS)

    df = df.sort_values("Priority_Gap", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.barh(
        df["County"],
        df["Priority_Gap"],
    )

    ax.set_title(
        "Mental Health Resource Priority Gap by County",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Priority Gap Score")
    ax.set_ylabel("County")

    for bar in bars:
        width = bar.get_width()

        ax.text(
            width + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fig_priority_gap_rankings.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_capacity_heatmap():
    df = pd.read_csv(CAPACITY_PROFILE)

    metrics = []

    for col in [
        "MH_Providers_per_100k",
        "Licensed_MH_Facilities_Count",
        "Psych_Beds_per_100k",
    ]:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")

            if numeric.notna().sum() > 0 and numeric.nunique() > 1:
                df[col] = numeric
                metrics.append(col)

    if len(metrics) == 0:
        print(
            "Skipping capacity heatmap: "
            "no usable varying numeric capacity metrics."
        )
        return

    heatmap_df = df.set_index("County")[metrics]

    normalized = (
        heatmap_df - heatmap_df.min()
    ) / (
        heatmap_df.max() - heatmap_df.min()
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        normalized,
        annot=heatmap_df.round(1),
        fmt="",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(
        "County Capacity Profile Heatmap",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fig_capacity_profile_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def create_facility_concentration_chart():
    if not FACILITY_CONCENTRATION.exists():
        print(
            "Skipping facility concentration chart: "
            "facility_concentration.csv not found."
        )
        return

    df = pd.read_csv(FACILITY_CONCENTRATION)

    if df.empty:
        print(
            "Skipping facility concentration chart: "
            "dataset is empty."
        )
        return

    df = df.sort_values(
        "Facility_Share_Pct",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.barh(
        df["County"],
        df["Facility_Share_Pct"],
    )

    ax.set_title(
        "Share of Metro Atlanta Mental Health Facilities",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xlabel("Facility Share (%)")

    for bar in bars:
        width = bar.get_width()

        ax.text(
            width + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fig_facility_concentration.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def create_mhpsa_chart():
    if not MHPSA_SUMMARY.exists():
        print(
            "Skipping MHPSA chart: "
            "mhpsa_summary.csv not found."
        )
        return

    df = pd.read_csv(MHPSA_SUMMARY)

    if df.empty:
        print(
            "Skipping MHPSA chart: "
            "dataset is empty."
        )
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.barh(
        df["MHPSA_Designation_Status"],
        df["County_Count"],
    )

    ax.set_title(
        "MHPSA Designation Counts",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xlabel("County Count")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fig_mhpsa_designations.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def create_capacity_profile_table():
    df = pd.read_csv(CAPACITY_PROFILE)

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "table_capacity_profile.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_priority_gap_table():
    df = pd.read_csv(PRIORITY_GAPS)

    top5 = df.head(5)[
        [
            "County",
            "Priority_Gap",
            "Need_Score",
            "MH_Providers_per_100k",
        ]
    ]

    fig, ax = plt.subplots(figsize=(8, 2.5))

    ax.axis("off")

    table = ax.table(
        cellText=top5.round(2).values,
        colLabels=top5.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "table_priority_gap_top5.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_psych_bed_summary_table():
    df = pd.read_csv(PSYCH_BED_SUMMARY)

    fig, ax = plt.subplots(figsize=(7, 2.8))

    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "table_psych_bed_summary.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    create_priority_gap_chart()
    create_capacity_heatmap()

    if FACILITY_CONCENTRATION.exists():
        create_facility_concentration_chart()

    if MHPSA_SUMMARY.exists():
        create_mhpsa_chart()

    create_capacity_profile_table()
    create_priority_gap_table()

    if PSYCH_BED_SUMMARY.exists():
        create_psych_bed_summary_table()

    print("\nVisualization generation complete.")

if __name__ == "__main__":
    main()