"""Reproduce Figure 9 (Morel 2-biome FOV) in the paper's visual style.

Reads the aggregated `data.parquet` and renders curves using the same colors,
line widths, legend layout, and spines as forager-agents/learning_curve.py.
Run from the repo root:

    python experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/plot_paper_style.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

try:
    from PyExpPlotting.matplot import save, setDefaultConference, setFonts

    setDefaultConference("jmlr")
    setFonts(20)
except ImportError:
    save = None
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        }
    )

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / EXP_DIR.relative_to(ROOT / "experiments")
PARQUET = RESULTS_DIR / "data.parquet"
PLOT_DIR = EXP_DIR / "plots"

COLORS = {
    "DQN-3":  "#00ffff",
    "DQN-5":  "#3ddcff",
    "DQN-7":  "#57abff",
    "DQN-9":  "#8b8cff",
    "DQN-11": "#b260ff",
    "DQN-13": "#d72dff",
    "DQN-15": "#ff00ff",
    "Search-Oracle":  "#34a853",  # GDMColor.GREEN
    "Random":         "#000000",  # GDMColor.BLACK
    "Search-Nearest": "#ea4335",  # GDMColor.RED
}

LABEL_MAP = {
    "Search-Oracle":  "Search Oracle",
    "Search-Nearest": "Search Nearest",
}

ORDER = {"Search-Oracle": 100, "Random": 101, "Search-Nearest": 102}

METRIC = "ewm_reward"


def alg_key(alg: str, aperture):
    if alg == "DQN":
        return f"DQN-{int(aperture)}"
    return alg


def sort_idx(key: str):
    if key.startswith("DQN-"):
        return int(key.split("-")[1])
    return ORDER[key]


def main():
    df = pl.read_parquet(PARQUET)
    # Cast Float16 columns to Float32 — Polars aggregations don't support Float16.
    df = df.with_columns(pl.col(pl.Float16).cast(pl.Float32))

    fig, ax = plt.subplots(figsize=(10, 6))

    keys = []
    for (alg, aperture), group_df in df.group_by(["alg", "aperture"]):
        key = alg_key(alg, aperture)
        if key not in COLORS:
            continue
        keys.append((key, group_df))

    keys.sort(key=lambda kv: sort_idx(kv[0]))

    for key, group_df in keys:
        # Average over seeds at each frame, ±1.96 SEM band
        agg = (
            group_df.group_by("frame")
            .agg(
                pl.col(METRIC).mean().alias("mean"),
                pl.col(METRIC).std().alias("std"),
                pl.col(METRIC).count().alias("n"),
            )
            .sort("frame")
            .drop_nulls()
        )

        frames = agg["frame"].to_numpy()
        means = agg["mean"].to_numpy()
        stds = agg["std"].to_numpy()
        ns = np.maximum(agg["n"].to_numpy(), 1)
        sem = stds / np.sqrt(ns)
        lower = means - 1.96 * sem
        upper = means + 1.96 * sem

        label = LABEL_MAP.get(key, key)
        ax.plot(frames, means, label=label, color=COLORS[key], linewidth=0.5)
        ax.fill_between(
            frames, lower, upper, color=COLORS[key], alpha=0.2, linewidth=0
        )

    ax.set_xlabel("Time steps")
    ax.set_ylabel("Average Reward")
    ax.set_ylim(0, 2)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if save is not None:
        save(
            save_path=str(PLOT_DIR),
            plot_name="learning_curve",
            save_type="pdf",
            width=1.2,
            height_ratio=1 / 1.2,
        )
        plt.clf()
    else:
        out = PLOT_DIR / "learning_curve.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved plot to {PLOT_DIR}/learning_curve.pdf")


if __name__ == "__main__":
    main()
