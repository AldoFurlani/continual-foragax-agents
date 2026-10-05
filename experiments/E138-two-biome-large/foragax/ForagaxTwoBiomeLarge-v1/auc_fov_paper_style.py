"""Reproduce the AUC-vs-FOV plot from forager-agents/auc_fov.py.

Last 10% average reward, per FOV. DQN gets a line varying with aperture;
baselines (Random, Search-Oracle, Search-Nearest) are constant horizontal
lines because they don't depend on aperture.

Run from the repo root:

    python experiments/E138-two-biome-large/foragax/ForagaxTwoBiomeLarge-v1/auc_fov_paper_style.py
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
        }
    )

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / EXP_DIR.relative_to(ROOT / "experiments")
PARQUET = RESULTS_DIR / "data.parquet"
PLOT_DIR = EXP_DIR / "plots"

COLORS = {
    "DQN":            "#4285f4",  # GDMColor.BLUE
    "Random":         "#000000",  # GDMColor.BLACK
    "Search-Oracle":  "#34a853",  # GDMColor.GREEN
    "Search-Nearest": "#ea4335",  # GDMColor.RED
}

METRIC = "ewm_reward"
LAST_PERCENT = 0.1
APERTURES = [3, 5, 7, 9, 11, 13, 15]


def bootstrap_mean_ci(values, n_boot=2000, ci=(2.5, 97.5), rng=None):
    """Bootstrap the mean of a 1D array. Returns (mean, lower, upper)."""
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(boot_means, ci[0])),
        float(np.percentile(boot_means, ci[1])),
    )


def main():
    df = pl.read_parquet(PARQUET)
    df = df.with_columns(pl.col(pl.Float16).cast(pl.Float32))

    # All runs are the same total_steps (500k), so use a global cutoff.
    # Per-group joins on aperture would drop baselines because aperture is null
    # for them and Polars doesn't match null==null in joins by default.
    max_frame_value = df["frame"].max()
    assert max_frame_value is not None, "frame column is empty"
    max_frame = int(max_frame_value)  # type: ignore[arg-type]
    df = df.filter(pl.col("frame") >= (1 - LAST_PERCENT) * max_frame)

    # Mean per (alg, aperture, seed) over the last-10% window.
    per_seed = (
        df.group_by(["alg", "aperture", "seed"])
        .agg(pl.col(METRIC).mean().alias("seed_mean"))
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    # DQN — one bootstrap stat per aperture.
    dqn_means, dqn_lo, dqn_hi = [], [], []
    for ap in APERTURES:
        vals = (
            per_seed.filter((pl.col("alg") == "DQN") & (pl.col("aperture") == ap))
            .select("seed_mean")
            .to_series()
            .to_numpy()
        )
        m, lo, hi = bootstrap_mean_ci(vals)
        dqn_means.append(m)
        dqn_lo.append(lo)
        dqn_hi.append(hi)

    ax.plot(APERTURES, dqn_means, color=COLORS["DQN"], linewidth=1)
    ax.fill_between(APERTURES, dqn_lo, dqn_hi, color=COLORS["DQN"], alpha=0.2)

    # Baselines — horizontal lines.
    for baseline in ["Search-Oracle", "Search-Nearest", "Random"]:
        vals = (
            per_seed.filter(pl.col("alg") == baseline)
            .select("seed_mean")
            .to_series()
            .to_numpy()
        )
        m, lo, hi = bootstrap_mean_ci(vals)
        if np.isnan(m):
            continue
        ax.plot(APERTURES, [m] * len(APERTURES), color=COLORS[baseline], linewidth=1)
        ax.fill_between(
            APERTURES,
            [lo] * len(APERTURES),
            [hi] * len(APERTURES),
            color=COLORS[baseline],
            alpha=0.4,
        )

    # Manual text labels (no legend, like the original). y values place the
    # text edge just past the CI band (`va` flips which edge `y` refers to).
    ax.text(3, 1.70, "Search Oracle", color=COLORS["Search-Oracle"], va="top")
    ax.text(15, 1.65, "DQN", color=COLORS["DQN"], ha="right", va="top")
    ax.text(15, 1.02, "Search Nearest", color=COLORS["Search-Nearest"], ha="right", va="bottom")
    ax.text(15, 0.58, "Random", color=COLORS["Random"], ha="right", va="top")

    ax.set_xlabel("Field of View")
    ax.set_ylabel("Last 10% Average Reward AUC")
    ax.set_xticks(APERTURES)
    ax.set_xticklabels([str(x) for x in APERTURES])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if save is not None:
        save(save_path=str(PLOT_DIR), plot_name="auc_fov", save_type="pdf")
        plt.clf()
    else:
        out = PLOT_DIR / "auc_fov.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved plot to {PLOT_DIR}/auc_fov.pdf")


if __name__ == "__main__":
    main()
