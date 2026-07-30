"""
visualize.py

Generates the three visuals used in the project write-up:
    1. mental_health_index_chart.png   - ranked bar chart of MHI scores
    2. need_vs_capacity_scatter.png    - bubble scatter: need vs. provider
                                          capacity, sized by mental distress,
                                          colored by MHI score
    3. mhi_choropleth_map.png / .html  - county-level choropleth of MHI scores

Requires: outputs/mental_health_index_final.csv to already exist
          (run build_index.py first).

Usage:
    python src/build_index.py
    python src/visualize.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from matplotlib.patches import Polygon

MHI_PATH = "../outputs/mental_health_index_final.csv"
GEOJSON_PATH = "../data/ga_metro_counties_geojson.json"

FIPS_BY_COUNTY = {
    "Fulton": "13121",
    "DeKalb": "13089",
    "Gwinnett": "13135",
    "Cobb": "13067",
    "Clayton": "13063",
    "Cherokee": "13057",
    "Henry": "13151",
    "Douglas": "13097",
    "Fayette": "13113",
    "Rockdale": "13247",
}


def plot_ranked_bar_chart(
    df: pd.DataFrame, out_path: str = "../outputs/mental_health_index_chart.png"
):
    """Horizontal bar chart of MHI scores, ranked low to high."""
    plot_df = df.sort_values("MHI_Score_0to100", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [
        "#c0392b" if v >= 60 else "#e67e22" if v >= 35 else "#27ae60"
        for v in plot_df["MHI_Score_0to100"]
    ]
    bars = ax.barh(plot_df["County"], plot_df["MHI_Score_0to100"], color=colors)
    ax.set_xlabel("Mental Health Index Score (0-100, higher = greater need)")
    ax.set_title(
        "Metro Atlanta Mental Health Index by County", fontsize=13, fontweight="bold"
    )
    for bar, val in zip(bars, plot_df["MHI_Score_0to100"]):
        ax.text(
            val + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val}",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, 105)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_need_vs_capacity_scatter(
    df: pd.DataFrame, out_path: str = "../outputs/need_vs_capacity_scatter.png"
):
    """
    Bubble scatter: x = socioeconomic need composite (poverty/uninsured/
    unemployment average), y = provider density, bubble size = frequent
    mental distress %, color = overall MHI score.
    """
    df = df.copy()
    df["Need_Composite"] = df[
        ["Poverty_Rate_Pct", "Uninsured_Rate_Under65_Pct", "Unemployment_Rate_Pct"]
    ].mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    sizes = (
        df["Frequent_Mental_Distress_Pct"]
        - df["Frequent_Mental_Distress_Pct"].min()
        + 2
    ) ** 2.6

    scatter = ax.scatter(
        df["Need_Composite"],
        df["MH_Providers_per_100k"],
        s=sizes,
        c=df["MHI_Score_0to100"],
        cmap="OrRd",
        edgecolor="#333333",
        linewidth=1.2,
        alpha=0.85,
    )

    for _, row in df.iterrows():
        ax.annotate(
            row["County"],
            (row["Need_Composite"], row["MH_Providers_per_100k"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    xmed = df["Need_Composite"].median()
    ymed = df["MH_Providers_per_100k"].median()
    ax.axvline(xmed, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(ymed, color="gray", linestyle="--", linewidth=0.8)

    ax.text(
        0.98,
        0.02,
        "HIGH NEED\nLOW CAPACITY\n(priority)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#c0392b",
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.98,
        "LOW NEED\nHIGH CAPACITY\n(well-positioned)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#27ae60",
        fontweight="bold",
    )

    ax.set_xlabel(
        "Socioeconomic Need Composite (avg. of poverty, uninsured, unemployment %)",
        fontsize=10,
    )
    ax.set_ylabel("Mental Health Providers per 100,000 residents", fontsize=10)
    ax.set_title(
        "Need vs. Capacity — Metro Atlanta Counties\n"
        "(bubble size = Frequent Mental Distress %, color = MHI Score)",
        fontsize=12,
        fontweight="bold",
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("MHI Score (0-100)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_choropleth_static(
    df: pd.DataFrame, geojson: dict, out_path: str = "../outputs/mhi_choropleth_map.png"
):
    """
    Static county choropleth built by drawing raw GeoJSON polygons with
    matplotlib. (We draw polygons manually rather than using a GIS library
    like geopandas to keep dependencies light; see requirements.txt.)
    """
    score_by_fips = {
        FIPS_BY_COUNTY[row.County]: row.MHI_Score_0to100 for row in df.itertuples()
    }

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = matplotlib.colormaps["OrRd"]
    norm = mcolors.Normalize(
        vmin=df["MHI_Score_0to100"].min(), vmax=df["MHI_Score_0to100"].max()
    )

    all_lons, all_lats = [], []
    centroids = {}

    for feat in geojson["features"]:
        fips = feat["id"]
        name = feat["properties"]["NAME"]
        score = score_by_fips[fips]
        color = cmap(norm(score))
        geom = feat["geometry"]

        rings = (
            [geom["coordinates"][0]]
            if geom["type"] == "Polygon"
            else [poly[0] for poly in geom["coordinates"]]
        )
        for ring in rings:
            poly_patch = Polygon(
                ring, closed=True, facecolor=color, edgecolor="#333333", linewidth=1.2
            )
            ax.add_patch(poly_patch)
            xs = [c[0] for c in ring]
            ys = [c[1] for c in ring]
            all_lons.extend(xs)
            all_lats.extend(ys)

        biggest_ring = max(rings, key=len)
        xs = [c[0] for c in biggest_ring]
        ys = [c[1] for c in biggest_ring]
        centroids[name] = (sum(xs) / len(xs), sum(ys) / len(ys))

    for name, (lon, lat) in centroids.items():
        score = score_by_fips[FIPS_BY_COUNTY[name]]
        ax.annotate(
            f"{name}\n{score}",
            (lon, lat),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    ax.set_xlim(min(all_lons) - 0.05, max(all_lons) + 0.05)
    ax.set_ylim(min(all_lats) - 0.05, max(all_lats) + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Metro Atlanta Mental Health Index (MHI) by County",
        fontsize=14,
        fontweight="bold",
    )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.04, pad=0.02)
    cbar.set_label("MHI Score (0-100, higher = greater need)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_choropleth_interactive(
    df: pd.DataFrame, geojson: dict, out_path: str = "../outputs/mhi_choropleth_map.html"
):
    """Interactive hover-enabled choropleth, saved as a standalone HTML file."""
    plot_df = df.copy()
    plot_df["fips"] = plot_df["County"].map(FIPS_BY_COUNTY)

    fig = px.choropleth(
        plot_df,
        geojson=geojson,
        locations="fips",
        color="MHI_Score_0to100",
        color_continuous_scale="OrRd",
        hover_name="County",
        hover_data={
            "fips": False,
            "MHI_Score_0to100": True,
            "Poverty_Rate_Pct": True,
            "Frequent_Mental_Distress_Pct": True,
            "MH_Providers_per_100k": True,
        },
        scope="usa",
        labels={"MHI_Score_0to100": "MHI Score"},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        title_text="Metro Atlanta Mental Health Index (MHI) by County",
        title_x=0.5,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="MHI Score<br>(0-100)"),
    )
    fig.write_html(out_path)
    print(f"Wrote {out_path}")


def main():
    df = pd.read_csv(MHI_PATH)
    with open(GEOJSON_PATH) as f:
        geojson = json.load(f)

    plot_ranked_bar_chart(df)
    plot_need_vs_capacity_scatter(df)
    plot_choropleth_static(df, geojson)
    plot_choropleth_interactive(df, geojson)


if __name__ == "__main__":
    main()
