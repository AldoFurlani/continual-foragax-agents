"""Plot T-BPTT results against the truncation window `seq_len`.

The window is the treatment variable of the E142 experiments, so the deliverable
is a curve over it rather than a single selected value: a rise that plateaus is
a mechanism claim (the effect tracks the credit-assignment horizon), whereas one
window beating a baseline is one number beating another.

Two panels:

  performance  -- late-window mean reward per window, with the matched
                  real-time (RTRL) agent as a horizontal reference when
                  --baseline-path is given.

  memory       -- the learned RTU pole magnitude `r`, reported as the memory
                  time constant tau = -1 / ln(r), against the window. This is
                  the more sensitive instrument: the loss can only supply
                  evidence for retention it can see, so tau should climb with
                  the window and then SATURATE once the window exceeds what the
                  task actually needs. Where it flattens estimates the task's
                  memory requirement. The dashed tau = T line marks the point
                  where the forward memory outruns the gradient window -- above
                  it, the network is using history it was never credited for
                  building.

Reads the eval parquet (one hyperparameter setting per agent), NOT a sweep
parquet: process_data's dedup key drops `id`, so a sweep parquet retains only
the last permutation per agent.

Usage:
    python src/seq_len_curve.py experiments/E142-bptt/foragax/ForagaxBig-v5
    python src/seq_len_curve.py experiments/E142-bptt/foragax/ForagaxSquareWaveTwoBiome-v11 \
        --arch MLP \
        --baseline-path experiments/E139-ppo-plasticity/foragax/ForagaxSquareWaveTwoBiome-v11 \
        --baseline-alg RealTimeActorCriticMLP
"""

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from plotting_utils import despine, load_data, save_plot

# BPTTActorCriticMLP_T16 -> ("MLP", 16)
# The Stacked variants must precede their bare counterparts in the alternation
# so the stacked agents are not read as the single-layer ones.
AGENT_RE = re.compile(
    r"^BPTTActorCritic(?P<arch>ConvStacked|Conv|MLPStacked|MLP)_T(?P<seq_len>\d+)$"
)

# Bounds on the reported memory time constant. tau = -1/ln(r) is unbounded as
# r -> 1, so cap it well above any horizon these environments contain (v11's
# void crossings and meal spacing are tens of steps) and report the clipping
# rather than plotting a meaningless 1e9.
R_MIN = 1e-6
R_MAX = 1.0 - 1e-4
TAU_MAX = -1.0 / math.log(R_MAX)  # ~1e4 steps


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="Experiment directory (the eval dir, not -sweep)")
    p.add_argument("--arch", choices=["Conv", "ConvStacked", "MLP", "MLPStacked"], default=None,
                   help="Restrict to one architecture (default: every arch present)")
    p.add_argument("--aperture", type=int, default=9)
    p.add_argument("--metric", default="ewm_reward")
    p.add_argument("--tail-fraction", type=float, default=0.25,
                   help="Fraction of the run, measured from the end, to average "
                        "over. Late-window so the comparison reflects converged "
                        "behaviour rather than early transients.")
    p.add_argument("--baseline-path", default=None,
                   help="Experiment dir holding the matched real-time agent")
    p.add_argument("--baseline-alg", default="RealTimeActorCriticMLP")
    p.add_argument("--baseline-end-frame", type=float, default=None,
                   help="Truncate the baseline to this many frames before taking "
                        "its late window. Needed when the baseline ran a longer "
                        "horizon than the T-BPTT agents: --tail-fraction is "
                        "relative to each run's own last frame, so a 30M "
                        "baseline would otherwise be summarised over 22.5M-30M "
                        "and compared against a 10M run's 7.5M-10M.")
    p.add_argument("--save-type", default="png")
    p.add_argument("--plot-name", default=None)
    return p.parse_args()


def _tail(df: pl.DataFrame, tail_fraction: float) -> pl.DataFrame:
    """Rows in the last `tail_fraction` of the run, by frame."""
    max_frame = df.select(pl.col("frame").max()).item()
    if max_frame is None:
        return df
    return df.filter(pl.col("frame") >= max_frame * (1.0 - tail_fraction))


def _per_seed(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Late-window mean of `col` for each seed; NaNs (inactive metric steps,
    e.g. r_stats only emits on its own cadence) are dropped, not zero-filled."""
    if col not in df.columns:
        return pl.DataFrame({"seed": [], col: []})
    return (
        df.filter(pl.col(col).is_not_null() & pl.col(col).is_not_nan())
        .group_by("seed")
        .agg(pl.col(col).mean().alias(col))
        .sort("seed")
    )


def _mean_ci(values: np.ndarray):
    """Mean and 95% normal-approximation CI half-width over seeds."""
    n = len(values)
    if n == 0:
        return math.nan, 0.0
    if n == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(1.96 * values.std(ddof=1) / math.sqrt(n))


def collect(df: pl.DataFrame, arch: str, aperture: int, metric: str,
            tail_fraction: float):
    """-> {seq_len: {"reward": (mean, ci), "tau": (mean, ci), "seeds": n}}"""
    out = {}
    for alg in sorted(df.select("alg").unique().to_series().to_list()):
        m = AGENT_RE.match(alg)
        if not m or m.group("arch") != arch:
            continue
        sub = df.filter((pl.col("alg") == alg) & (pl.col("aperture") == aperture))
        if sub.height == 0:
            continue
        sub = _tail(sub, tail_fraction)

        rewards = _per_seed(sub, metric)[metric].to_numpy()
        entry = {"reward": _mean_ci(rewards), "seeds": len(rewards)}

        r_vals = _per_seed(sub, "rtu_r_mean")
        if r_vals.height:
            raw = r_vals["rtu_r_mean"].to_numpy()
            # tau = -1/ln(r) diverges as r -> 1, so a pole that has learned its
            # way to the stability boundary would silently render as tau ~ 1e9
            # and flatten the log axis. Cap at R_MAX and say so: "saturated" is
            # the finding, not a number to plot.
            n_sat = int((raw >= R_MAX).sum())
            if n_sat:
                print(f"  ! {alg}: {n_sat}/{len(raw)} seeds have r >= {R_MAX} "
                      f"(tau capped at {TAU_MAX:.0f} steps; memory is at the "
                      f"stability boundary, not measurable from r alone)")
            r = np.clip(raw, R_MIN, R_MAX)
            entry["tau"] = _mean_ci(-1.0 / np.log(r))
            entry["saturated"] = n_sat
        out[int(m.group("seq_len"))] = entry
    return out


def baseline_value(args):
    """Late-window mean of the matched real-time agent, or None."""
    if not args.baseline_path:
        return None
    df = load_data(Path(args.baseline_path).resolve())
    sub = df.filter(
        (pl.col("alg") == args.baseline_alg) & (pl.col("aperture") == args.aperture)
    )
    if sub.height == 0:
        print(f"  ! baseline {args.baseline_alg} not found in {args.baseline_path}")
        return None
    if args.baseline_end_frame is not None:
        sub = sub.filter(pl.col("frame") <= args.baseline_end_frame)
        if sub.height == 0:
            print(f"  ! baseline {args.baseline_alg} has no frames <= "
                  f"{args.baseline_end_frame:.0f}")
            return None
        print(f"  baseline {args.baseline_alg} truncated to "
              f"{args.baseline_end_frame:.0f} frames before its late window")
    vals = _per_seed(_tail(sub, args.tail_fraction), args.metric)[args.metric].to_numpy()
    return _mean_ci(vals)


def plot_arch(arch, points, args, baseline):
    windows = sorted(points)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    means = np.array([points[w]["reward"][0] for w in windows])
    cis = np.array([points[w]["reward"][1] for w in windows])
    ax.errorbar(windows, means, yerr=cis, marker="o", capsize=4, lw=2,
                label="T-BPTT", color="#4477AA")
    if baseline is not None:
        b_mean, b_ci = baseline
        ax.axhline(b_mean, ls="--", lw=2, color="#CC6677",
                   label=f"real-time ({args.baseline_alg})")
        ax.fill_between([min(windows), max(windows)], b_mean - b_ci, b_mean + b_ci,
                        color="#CC6677", alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_xticks(windows)
    ax.set_xticklabels([str(w) for w in windows])
    ax.set_xlabel("truncation window T (seq_len)")
    ax.set_ylabel(f"late-window {args.metric}")
    ax.set_title(f"{arch}: performance vs window")
    ax.legend(frameon=False, fontsize=9)
    despine(ax)

    ax = axes[1]
    have_tau = [w for w in windows if "tau" in points[w]]
    if have_tau:
        tmeans = np.array([points[w]["tau"][0] for w in have_tau])
        tcis = np.array([points[w]["tau"][1] for w in have_tau])
        ax.errorbar(have_tau, tmeans, yerr=tcis, marker="s", capsize=4, lw=2,
                    color="#228833", label=r"learned $\tau = -1/\ln r$")
        ax.plot(have_tau, have_tau, ls="--", lw=1.5, color="grey",
                label=r"$\tau = T$")
        ax.set_yscale("log", base=2)
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.text(0.5, 0.5, "no rtu_r_mean in results\n(set experiment.r_stats_freq)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_xticks(windows)
    ax.set_xticklabels([str(w) for w in windows])
    ax.set_xlabel("truncation window T (seq_len)")
    ax.set_ylabel(r"memory time constant $\tau$ (steps)")
    ax.set_title(f"{arch}: learned memory vs window")
    despine(ax)

    fig.tight_layout()
    name = args.plot_name or f"seq_len_curve_{arch}"
    save_plot(fig, Path(args.path).resolve(), name, args.save_type)
    plt.close(fig)
    return name


def main():
    args = parse_args()
    exp_path = Path(args.path).resolve()
    df = load_data(exp_path)

    archs = [args.arch] if args.arch else ["Conv", "ConvStacked", "MLP", "MLPStacked"]
    baseline = baseline_value(args)

    for arch in archs:
        points = collect(df, arch, args.aperture, args.metric, args.tail_fraction)
        if not points:
            print(f"{arch}: no BPTTActorCritic{arch}_T* results found, skipping")
            continue
        for w in sorted(points):
            e = points[w]
            tau = f", tau={e['tau'][0]:.1f}" if "tau" in e else ""
            print(f"{arch} T={w:<3} {args.metric}={e['reward'][0]:.4f}"
                  f" +/- {e['reward'][1]:.4f}  (n={e['seeds']} seeds){tau}")
        name = plot_arch(arch, points, args, baseline)
        print(f"{arch}: wrote plots/{name}.{args.save_type}")


if __name__ == "__main__":
    main()
