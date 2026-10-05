"""Plot a learning curve for a single experiment / agent / FOV.

A single "do-it-all" replacement for the per-experiment ``plot_single_fov.py``
copies.  Point it at any experiment (the ``experiments/...`` path, matching the
``-e experiments/...`` convention used elsewhere in this codebase), pick an
agent and FOV, and it reads that experiment's ``data.parquet``, averages over
seeds, and writes the plot into that experiment's ``plots/`` folder.

Examples:
    python scripts/plot_single_fov.py \
        -e experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 \
        -a ActorCriticMLP -f 9

    python scripts/plot_single_fov.py \
        -e experiments/E136-big/foragax/ForagaxBig-v5 -a DQN -f 9 --ylim 0 0.05
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> repo root
sys.path.insert(0, str(ROOT / "src"))

# Known per-agent colors; unlisted agents fall back to the matplotlib cycle.
COLORS = {
    "DQN": "#8b8cff",
    "ActorCriticMLP": "#ff8c42",
    "RealTimeActorCriticMLP": "#42a5ff",
    "RealTimeActorCriticConv": "#3ca370",
    "ActorCriticConv": "#d65db1",
}


def _apply_style():
    try:
        from PyExpPlotting.matplot import setDefaultConference, setFonts

        setDefaultConference("jmlr")
        setFonts(20)
    except ImportError:
        plt.rcParams.update({
            "font.size": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        })


def _normalize_exp(exp: str) -> str:
    """Strip a leading experiments/ or results/ prefix and trailing slash."""
    exp = exp.strip().rstrip("/")
    for prefix in ("experiments/", "results/"):
        if exp.startswith(prefix):
            exp = exp[len(prefix):]
    return exp


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "-e", "--exp", required=True,
        help="Experiment path, e.g. experiments/X33-ForagaxSquareWaveTwoBiome-v11/"
             "foragax/ForagaxSquareWaveTwoBiome-v11 (a leading experiments/ or "
             "results/ is optional; both forms work)",
    )
    p.add_argument("-a", "--agent", required=True,
                   help="Agent / algorithm name as stored in the 'alg' column (e.g. DQN, ActorCriticMLP)")
    p.add_argument("-f", "--fov", required=True, type=int,
                   help="Field-of-view / aperture size (the 'aperture' column)")
    p.add_argument("-m", "--metric", default="ewm_reward",
                   help="Column to plot on the y-axis (default: ewm_reward)")
    p.add_argument("--ylim", nargs=2, type=float, metavar=("YMIN", "YMAX"),
                   default=None, help="Fixed y-axis limits; default autoscales")
    p.add_argument("--ylabel", default="Average Reward", help="Y-axis label")
    return p.parse_args()


def main():
    args = parse_args()
    _apply_style()

    exp = _normalize_exp(args.exp)
    parquet = ROOT / "results" / exp / "data.parquet"
    plot_dir = ROOT / "experiments" / exp / "plots"

    if not parquet.exists():
        print(f"Parquet file not found: {parquet}")
        return

    df = pl.read_parquet(parquet)
    # Cast Float16 columns to Float32 — Polars aggregations don't support Float16.
    df = df.with_columns(pl.col(pl.Float16).cast(pl.Float32))

    for col in ("alg", "aperture", "frame", args.metric):
        if col not in df.columns:
            print(f"Column '{col}' not found in {parquet.name}. "
                  f"Available: {df.columns}")
            return

    filtered_df = df.filter(
        (pl.col("aperture") == args.fov) & (pl.col("alg") == args.agent)
    )
    if filtered_df.height == 0:
        print(f"No data for agent={args.agent!r} FOV={args.fov}.")
        print(f"  Available agents: {df['alg'].unique().to_list()}")
        print(f"  Available FOVs:   {df['aperture'].unique().to_list()}")
        return

    # Average over seeds at each frame, ±1.96 SEM band.
    agg = (
        filtered_df.group_by("frame")
        .agg(
            pl.col(args.metric).mean().alias("mean"),
            pl.col(args.metric).std().alias("std"),
            pl.col(args.metric).count().alias("n"),
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

    fig, ax = plt.subplots(figsize=(10, 6))

    color = COLORS.get(args.agent)  # None -> matplotlib picks a default
    label = f"{args.agent}-{args.fov}"
    ax.plot(frames, means, label=label, color=color, linewidth=0.5)
    ax.fill_between(frames, lower, upper, color=color, alpha=0.2, linewidth=0)

    ax.set_xlabel("Time steps")
    ax.set_ylabel(args.ylabel)
    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"learning_curve_{args.agent}_{args.fov}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
