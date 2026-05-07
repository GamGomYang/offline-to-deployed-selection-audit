from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "package_c_interface_heatmap.csv"
OUTPUT_PDF = HERE / "fig_traffic_topk_heatmap.pdf"


def winner_label(model: str) -> str:
    return {
        "reactive_short": "R",
        "lagged_smoother": "S",
        "calibrated_baseline": "C",
    }.get(str(model), str(model)[:1].upper())


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    pivot = (
        df.pivot(index="k", columns="kappa", values="deployed_suboptimal_share")
        .sort_index()
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)

    ax.set_title("Traffic Top-$k$: share of deployed-suboptimal units", pad=12)
    ax.set_xlabel("Friction $\\kappa$", labelpad=11)
    ax.set_ylabel("Top-$k$ budget", labelpad=8)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:.2f}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index])

    for i, k in enumerate(pivot.index):
        for j, kappa in enumerate(pivot.columns):
            row = df[df["k"].eq(k) & np.isclose(df["kappa"], float(kappa))].iloc[0]
            label = winner_label(row["deployed_winner"])
            if bool(row["canonical_marker"]):
                label += "*"
            count = f"{int(row['deployed_suboptimal_count'])}/{int(row['n_units'])}"
            color = "white" if float(row["deployed_suboptimal_share"]) >= 0.45 else "black"
            ax.text(j, i, f"{label}\n{count}", ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Deployed-suboptimal share", labelpad=8)

    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.20, top=0.86)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
