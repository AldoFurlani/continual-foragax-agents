"""Per-layer actor-vs-critic plasticity comparison plots.

For each plasticity-metric family (effective rank, tanh saturation rate, Sokar
dormant fraction) this draws one figure with a subplot per probe *layer*, and
within each subplot the actor and critic curves overlaid (mean + bootstrap 95%
CI over seeds). This is the comparison the generic ``learning_curve.py`` can't
make directly, because there the hue is the algorithm, not the actor/critic
role.

Effective rank is normalized by the layer's activation width (``eff_rank / dim``)
so the three layers — pre1 (64), RTU (2*d_hidden), pre2 (64) — are on the same
[0,1] "fraction of available dimensions" scale and the wide RTU layer doesn't
swamp the axis. Saturation and dormant fractions are already in [0,1].

Three modes (``--mode``):
  absolute     metric vs wall-clock frame (``--sample-type every`` or a zoom
               window like ``4900000:5100000:500``).
  fold         switch-triggered average: re-index time to signed steps from the
               nearest switch (tau<0 before, tau=0 at, tau>0 after) and average
               over every switch in one dense window (``--window``) and over
               seeds. The principled "average effect of a switch", replacing an
               arbitrary single-switch zoom. ``--pre``/``--post`` crop the axis.
  fold-overlay fold several training-stage windows (``--windows``) and overlay
               them (early/mid/late), so a deepening/slowing post-switch
               transient over training — i.e. plasticity loss — is visible.

Switches alternate biome direction, but ForagaxSquareWaveTwoBiome is symmetric
by construction (the two half-periods have identical reward magnitudes with
roles swapped), so folding on the 250k half-period is valid.

Usage:
    python src/plasticity_compare.py <experiment_path>            # absolute, full
    python src/plasticity_compare.py <experiment_path> --mode fold
    python src/plasticity_compare.py <experiment_path> --mode fold-overlay
        [--alg RealTimeActorCriticMLP] [--families eff_rank sat_rate dormant]
        [--save-type pdf]
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import polars as pl

# Switches every 250k steps (square wave: half of the 500k period).
SWITCH_PERIOD = 250_000

ROLE_COLORS = {"actor": "tab:blue", "critic": "tab:red"}

# Training-stage colors for the fold-overlay (cool early -> warm late, to read
# as "the response changes / degrades over training").
STAGE_COLORS = ["#4575b4", "#fdae61", "#d73027"]

# Probe layers per metric family, with a human-readable title. Widths (for the
# eff_rank normalization) are filled in from the run config at load time.
FAMILIES = {
    "eff_rank": {
        "ylabel": "Effective rank / width",
        "normalize_by_width": True,
        "layers": [("pre1", "Pre-tanh 1"), ("rtu", "RTU output"), ("pre2", "Pre-tanh 2")],
    },
    "sat_rate": {
        "ylabel": "Tanh saturation rate",
        "normalize_by_width": False,
        "layers": [("pre1", "Pre-tanh 1"), ("pre2", "Pre-tanh 2")],
    },
    "sat_persist": {
        "ylabel": "Persistent saturation (|EMA tanh| > $\\tau$)",
        "normalize_by_width": False,
        "layers": [("pre1", "Pre-tanh 1"), ("pre2", "Pre-tanh 2")],
    },
    "dormant": {
        "ylabel": "Dormant fraction ($\\tau$=0.025)",
        "normalize_by_width": False,
        # rtu for the tanh RTU net; pre1/rtu/pre2 for the all-ReLU net (the
        # pre1/pre2 columns are simply absent for the tanh net and skipped).
        "layers": [("pre1", "Pre-tanh 1"), ("rtu", "RTU output"), ("pre2", "Pre-tanh 2")],
    },
    "dormant_persist": {
        "ylabel": "Persistent dormancy (EMA, score$\\leq$0.025)",
        "normalize_by_width": False,
        "layers": [("pre1", "Pre-tanh 1"), ("rtu", "RTU output"), ("pre2", "Pre-tanh 2")],
    },
}


def _parquet_path(experiment_path: Path) -> Path:
    """Map an experiments/... path to its results/.../data.parquet."""
    if experiment_path.suffix == ".parquet":
        return experiment_path
    parts = experiment_path.parts
    base = (
        Path("results", *parts[parts.index("experiments") + 1 :])
        if "experiments" in parts
        else experiment_path
    )
    return base / "data.parquet"


def _agent_config(experiment_path: Path, agent: str) -> dict:
    """The config json whose filename matches this alg (the alg == config stem),
    or {} if not found. Used to read the architecture and activation off the
    config rather than guessing from the name."""
    base = experiment_path.parent if experiment_path.suffix == ".parquet" else experiment_path
    for cfg in glob.glob(str(base / "**" / f"{agent}.json"), recursive=True):
        try:
            return json.loads(Path(cfg).read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return {}


def _resolve_activation(experiment_path: Path, agent: str, df) -> str:
    """tanh vs relu for an alg, resolved from the data first, then the config,
    then the filename suffix as a last resort:
      1. the parquet's `representation.activation` column (present for runs
         collected once activation became a config field),
      2. the matching config's representation.activation,
      3. the `…ReLU` filename suffix (older results predate the field)."""
    col = "representation.activation"
    if col in df.columns:
        vals = df[col].drop_nulls().unique().to_list()
        if len(vals) == 1:
            return str(vals[0])
    rep = _agent_config(experiment_path, agent).get("metaParameters", {}).get("representation", {})
    if rep.get("activation"):
        return str(rep["activation"])
    return "relu" if agent.endswith("ReLU") else "tanh"


def _resolve_is_rtu(experiment_path: Path, agent: str) -> bool:
    """Recurrent (RTU) vs feedforward MLP, from the config's top-level `agent`
    field (the network class), falling back to the filename prefix."""
    cls = _agent_config(experiment_path, agent).get("agent", agent)
    return cls.startswith("RealTime")


def _layer_widths(experiment_path: Path, activation: str, is_rtu: bool) -> dict:
    """Read hidden / d_hidden from the run config and derive eff_rank denominators.

    pre1/pre2 are Dense(hidden_size) for both architectures, normalized by width.
    The "rtu" slot differs by architecture:
      - RTU: nonlinear+recurrent output of width 2*d_hidden, which it can actually
        fill, so we normalize by that width.
      - vanilla (tanh) ActorCriticMLP: a *linear* Dense(d_hidden) whose rank is
        capped by the upstream ~hidden-wide bottleneck, NOT by its 512 width.
        Dividing by 512 would fabricate a collapse, so we return None -> plot raw
        rank there.
      - ReLU ActorCriticMLP: the wide Dense(d_hidden) is ReLU'd, so it can fill
        its own width -> normalize by d_hidden (NOT 2*d_hidden).
    Reads hidden/d_hidden from any config json; falls back to E139 defaults.
    """
    hidden, d_hidden = 64, 512
    cfgs = glob.glob(str(experiment_path / "**" / "*.json"), recursive=True)
    for cfg in cfgs:
        try:
            rep = json.loads(Path(cfg).read_text())["metaParameters"]["representation"]
            hidden, d_hidden = int(rep["hidden"]), int(rep["d_hidden"])
            break
        except (KeyError, json.JSONDecodeError, OSError):
            continue
    if is_rtu:
        rtu_denom = 2 * d_hidden  # RTU output width (tanh or relu)
    elif activation == "relu":
        rtu_denom = d_hidden  # ReLU'd wide Dense fills its own width
    else:
        rtu_denom = None  # vanilla tanh MLP: linear wide layer, plot raw rank
    return {"pre1": hidden, "pre2": hidden, "rtu": rtu_denom}


def _layer_titles(activation: str, is_rtu: bool) -> dict:
    """Architecture-aware panel titles. The pre1/pre2 sites are pre-tanh in the
    tanh nets and post-ReLU in the all-ReLU net; the wide slot is the linear
    Dense for vanilla, else the RTU output."""
    if is_rtu:
        mid = "RTU output"
    elif activation == "relu":
        mid = "Wide layer (ReLU)"
    else:
        mid = "Wide layer (linear)"
    if activation == "relu":
        return {"pre1": "ReLU 1", "pre2": "ReLU 2", "rtu": mid}
    return {"pre1": "Pre-tanh 1", "pre2": "Pre-tanh 2", "rtu": mid}


def _families_for(activation: str, is_rtu: bool) -> list:
    """Metric families that are meaningful for the architecture.
      - all-ReLU (RTU or MLP): dormancy (+ persistent dormancy) at every layer;
        no tanh, so no saturation families.
      - vanilla (tanh) MLP: tanh saturation (+ persistent); no Sokar dormancy
        (the wide layer is linear).
      - tanh RTU: saturation at the tanh sites, dormancy at the RTU output."""
    if activation == "relu":
        return ["eff_rank", "dormant", "dormant_persist"]
    if not is_rtu:
        return ["eff_rank", "sat_rate", "sat_persist"]
    return ["eff_rank", "sat_rate", "dormant", "sat_persist"]


def _present_layers(family, cfg, df):
    """cfg layers that actually have a column for this agent/data, so families
    whose layer set is wider than a given net (e.g. dormant: pre1/rtu/pre2 for
    ReLU but rtu-only for the tanh RTU) don't render empty panels."""
    return [
        (layer, title)
        for (layer, title) in cfg["layers"]
        if any(f"{family}_{role}_{layer}" in df.columns for role in ("actor", "critic"))
    ]


def _mean_ci(values_by_seed: np.ndarray, n_boot: int = 1000):
    """(mean, lo, hi) across the seed axis (axis 0); bootstrap 95% CI."""
    mean = np.nanmean(values_by_seed, axis=0)
    n = values_by_seed.shape[0]
    if n < 2:
        return mean, None, None
    rng = np.random.default_rng(0)
    boot = np.nanmean(values_by_seed[rng.integers(0, n, size=(n_boot, n))], axis=1)
    return mean, np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)


def _series(df: pl.DataFrame, col: str):
    """(frames, seed-major value matrix) for one metric column. aggregate_function
    guards against duplicate (seed, frame) rows (e.g. a doubled parquet)."""
    piv = df.pivot(
        values=col, index="seed", on="frame", aggregate_function="mean"
    ).sort("seed")
    frame_cols = sorted((c for c in piv.columns if c != "seed"), key=lambda c: int(c))
    mat = piv.select(frame_cols).to_numpy()
    frames = np.array([int(c) for c in frame_cols])
    return frames, mat


def _fold_series(df: pl.DataFrame, col: str, period: int = SWITCH_PERIOD):
    """Switch-triggered fold of one metric column, centered on the switch.

    Re-indexes time to tau = signed steps to the *nearest* switch (switches sit
    at multiples of `period`), so tau spans roughly [-period/2, +period/2): tau<0
    is the mature plateau approaching a switch, tau=0 is the switch, tau>0 is the
    fresh-biome transient after it. Centering this way keeps the pre-switch level
    on screen next to tau=0, so a collapse *at* the switch is visible (folding
    forward-only, tau = frame % period, hides it at the far edge).

    For each seed the metric is first averaged over the epochs at each tau
    (within-seed mean), giving one curve per seed; the caller then takes mean + CI
    across seeds, so the random unit is the seed (epochs within a seed are
    correlated and shouldn't be counted as independent). Returns
    (taus, seed-major matrix), or (None, None) if empty.
    """
    sub = df.select(["seed", "frame", col]).filter(pl.col(col).is_not_null())
    if sub.height == 0:
        return None, None
    half = period // 2
    # tau = frame - nearest_switch, where nearest_switch = round(frame/period)*period.
    sub = sub.with_columns(
        (pl.col("frame") - ((pl.col("frame") + half) // period) * period).alias("tau")
    )
    g = sub.group_by(["seed", "tau"]).agg(pl.col(col).mean().alias("v"))
    piv = g.pivot(values="v", index="seed", on="tau").sort("seed")
    tau_cols = sorted((c for c in piv.columns if c != "seed"), key=lambda c: int(c))
    mat = piv.select(tau_cols).to_numpy()
    taus = np.array([int(c) for c in tau_cols])
    return taus, mat


def _eff_denom(cfg, widths, layer):
    """eff_rank normalization denominator for a layer, or None to plot raw."""
    return widths.get(layer) if cfg["normalize_by_width"] else None


def plot_family(df, family, cfg, widths, titles, out_dir, sample_type, save_type):
    layers = _present_layers(family, cfg, df)
    fig, axes = plt.subplots(
        1, len(layers), figsize=(5.2 * len(layers), 4.0), squeeze=False
    )
    axes = axes[0]

    for ax, (layer, default_title) in zip(axes, layers, strict=True):
        denom = _eff_denom(cfg, widths, layer)
        for role in ("actor", "critic"):
            col = f"{family}_{role}_{layer}"
            if col not in df.columns:
                continue
            sub = df.select(["seed", "frame", col]).filter(pl.col(col).is_not_null())
            if sub.height == 0:
                continue
            frames, mat = _series(sub, col)
            if denom:
                mat = mat / denom
            x = frames / 1e6
            mean, lo, hi = _mean_ci(mat)
            ax.plot(x, mean, color=ROLE_COLORS[role], label=role, lw=1.6)
            if lo is not None:
                ax.fill_between(x, lo, hi, color=ROLE_COLORS[role], alpha=0.15)

        # Environment-switch markers inside the visible window. Dotted, thin and
        # faint so the ~39 markers over a full run read as switch lines without
        # turning into a dense field of dots.
        fmin, fmax = df["frame"].min(), df["frame"].max()
        first = ((fmin // SWITCH_PERIOD) + 1) * SWITCH_PERIOD
        for sw in range(int(first), int(fmax) + 1, SWITCH_PERIOD):
            ax.axvline(sw / 1e6, color="grey", linestyle=":", lw=0.6, alpha=0.4)

        ax.set_title(titles.get(layer, default_title))
        ax.set_xlabel(r"Time steps ($\times 10^6$)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.spines[["top", "right"]].set_visible(False)
        # A layer that should be width-normalized but can't be (vanilla linear
        # wide layer) is shown raw — label it so the axis isn't misread.
        if cfg["normalize_by_width"] and denom is None:
            ax.set_ylabel("Effective rank (raw)")
    axes[0].set_ylabel(cfg["ylabel"])
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()

    tag = "full" if sample_type == "every" else sample_type.replace(":", "-")
    out = out_dir / f"plasticity_compare_{family}_{tag}.{save_type}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _switch_line(ax):
    """Mark the environment switch at tau = 0."""
    ax.axvline(0.0, color="0.25", linestyle="--", lw=1.1, alpha=0.8, zorder=0)


def plot_fold_single(df, family, cfg, widths, titles, out_dir, window, save_type,
                     pre=None, post=None):
    """Canonical switch-triggered average from a single window: per layer, actor
    vs critic, averaged over all switches in the window and over seeds."""
    layers = _present_layers(family, cfg, df)
    fig, axes = plt.subplots(
        1, len(layers), figsize=(5.2 * len(layers), 4.0), squeeze=False
    )
    axes = axes[0]

    for ax, (layer, default_title) in zip(axes, layers, strict=True):
        denom = _eff_denom(cfg, widths, layer)
        for role in ("actor", "critic"):
            col = f"{family}_{role}_{layer}"
            if col not in df.columns:
                continue
            taus, mat = _fold_series(df, col)
            if taus is None or mat is None:
                continue
            if denom:
                mat = mat / denom
            x = taus / 1e3
            mean, lo, hi = _mean_ci(mat)
            ax.plot(x, mean, color=ROLE_COLORS[role], label=role, lw=1.6)
            if lo is not None:
                ax.fill_between(x, lo, hi, color=ROLE_COLORS[role], alpha=0.15)
        _switch_line(ax)
        if pre is not None or post is not None:
            ax.set_xlim(-(pre or 0) / 1e3, (post or 0) / 1e3)
        ax.set_title(titles.get(layer, default_title))
        ax.set_xlabel(r"Steps relative to switch ($\times 10^3$)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.spines[["top", "right"]].set_visible(False)
        if cfg["normalize_by_width"] and denom is None:
            ax.set_ylabel("Effective rank (raw)")
    axes[0].set_ylabel(cfg["ylabel"])
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(f"Switch-triggered average ({window})", fontsize=11)
    fig.tight_layout()

    out = out_dir / f"plasticity_fold_{family}_{window.replace(':', '-')}.{save_type}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_fold_overlay(df_all, family, cfg, widths, titles, out_dir, windows, labels,
                      save_type, pre=None, post=None):
    """Switch-triggered averages from several training-stage windows overlaid on
    one tau axis. Actor and critic on separate rows; one color per stage so a
    deepening / slowing transient over training (plasticity loss) is visible."""
    layers = _present_layers(family, cfg, df_all)
    roles = ("actor", "critic")
    colors = (
        STAGE_COLORS
        if len(windows) == len(STAGE_COLORS)
        else [plt.get_cmap("viridis")(t) for t in np.linspace(0.0, 0.85, len(windows))]
    )
    fig, axes = plt.subplots(
        len(roles), len(layers),
        figsize=(5.2 * len(layers), 3.6 * len(roles)), squeeze=False,
    )

    for r, role in enumerate(roles):
        for c, (layer, default_title) in enumerate(layers):
            ax = axes[r][c]
            denom = _eff_denom(cfg, widths, layer)
            for win, lab, color in zip(windows, labels, colors, strict=True):
                wdf = df_all.filter(pl.col("sample_type") == win)
                col = f"{family}_{role}_{layer}"
                if wdf.height == 0 or col not in wdf.columns:
                    continue
                taus, mat = _fold_series(wdf, col)
                if taus is None or mat is None:
                    continue
                if denom:
                    mat = mat / denom
                x = taus / 1e3
                mean, lo, hi = _mean_ci(mat)
                ax.plot(x, mean, color=color, label=lab, lw=1.6)
                if lo is not None:
                    ax.fill_between(x, lo, hi, color=color, alpha=0.15)
            _switch_line(ax)
            if pre is not None or post is not None:
                ax.set_xlim(-(pre or 0) / 1e3, (post or 0) / 1e3)
            if r == 0:
                # Mark a raw (un-normalized) eff_rank panel in the title.
                raw = cfg["normalize_by_width"] and denom is None
                ax.set_title(titles.get(layer, default_title) + (" (raw rank)" if raw else ""))
            if r == len(roles) - 1:
                ax.set_xlabel(r"Steps relative to switch ($\times 10^3$)")
            if c == 0:
                ax.set_ylabel(f"{role}\n{cfg['ylabel']}")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(frameon=False, loc="best", title="training stage")
    fig.tight_layout()

    out = out_dir / f"plasticity_fold_overlay_{family}.{save_type}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_path", type=str)
    parser.add_argument(
        "--mode",
        choices=["absolute", "fold", "fold-overlay"],
        default="absolute",
        help="absolute: metric vs wall-clock frame. fold: switch-triggered "
             "average from one window (--window). fold-overlay: fold each of "
             "--windows and overlay them by training stage.",
    )
    parser.add_argument(
        "--sample-type", type=str, default="every",
        help="Frame window for --mode absolute (e.g. 'every' or '4900000:5100000:500').",
    )
    parser.add_argument(
        "--window", type=str, default="4500000:5500000:500",
        help="Window to fold for --mode fold (a dense, multi-switch sample_type).",
    )
    parser.add_argument(
        "--windows", nargs="+",
        default=["1000000:2000000:500", "4500000:5500000:500", "9000000:10000000:500"],
        help="Windows to fold and overlay for --mode fold-overlay.",
    )
    parser.add_argument(
        "--window-labels", nargs="+", default=["early", "mid", "late"],
        help="Legend labels for --windows (must match in length).",
    )
    parser.add_argument(
        "--pre", type=int, default=None,
        help="Fold modes: steps to show before the switch (x-axis lower = -pre). "
             "Default shows the full half-period (~125k).",
    )
    parser.add_argument(
        "--post", type=int, default=None,
        help="Fold modes: steps to show after the switch (x-axis upper = +post). "
             "Default shows the full half-period (~125k).",
    )
    parser.add_argument("--alg", type=str, default=None)
    parser.add_argument(
        "--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES)
    )
    parser.add_argument("--save-type", type=str, default="pdf")
    args = parser.parse_args()

    if args.mode == "fold-overlay" and len(args.windows) != len(args.window_labels):
        sys.exit("--windows and --window-labels must have the same length.")

    exp = Path(args.experiment_path)
    pq = _parquet_path(exp)
    if not pq.exists():
        sys.exit(f"No parquet at {pq} — run process_data.py first.")
    df_all = pl.read_parquet(pq)
    if args.alg:
        df_all = df_all.filter(pl.col("alg") == args.alg)
        if df_all.height == 0:
            print(f"No rows for alg {args.alg} — skipping.")
            return
    elif df_all["alg"].n_unique() > 1:
        sys.exit(f"Multiple algs present {df_all['alg'].unique().to_list()}; pass --alg.")

    # Architecture: prefer the explicit --alg, else the single alg in the data.
    agent = args.alg or df_all["alg"][0]
    # Resolve activation (data -> config -> filename) and architecture (config
    # agent field -> filename) instead of inferring both from the alg string.
    activation = _resolve_activation(exp, agent, df_all)
    is_rtu = _resolve_is_rtu(exp, agent)
    widths = _layer_widths(exp, activation, is_rtu)
    titles = _layer_titles(activation, is_rtu)
    # Drop families that aren't meaningful for this architecture (vanilla has no
    # post-ReLU site, hence no dormancy), preserving the requested order.
    meaningful = _families_for(activation, is_rtu)
    families = [f for f in args.families if f in meaningful]
    dropped = [f for f in args.families if f not in meaningful]
    if dropped:
        print(f"Skipping families not defined for {agent}: {dropped}")
    # Skip families with no columns in the parquet yet (e.g. sat_persist before a
    # re-run that collects it) so we don't emit empty figures.
    def _present(fam):
        return any(
            f"{fam}_{role}_{layer}" in df_all.columns
            for role in ("actor", "critic")
            for layer, _ in FAMILIES[fam]["layers"]
        )
    absent = [f for f in families if not _present(f)]
    if absent:
        print(f"Skipping families with no data in the parquet: {absent}")
    families = [f for f in families if _present(f)]
    # One subfolder per algorithm so PPO and RTU figures don't mix.
    plots_dir = (exp.parent if exp.suffix == ".parquet" else exp) / "plots" / agent

    if args.mode == "fold-overlay":
        present = set(df_all["sample_type"].unique().to_list())
        missing = [w for w in args.windows if w not in present]
        if missing:
            sys.exit(f"Windows not found in parquet: {missing}")
        for family in families:
            plot_fold_overlay(
                df_all, family, FAMILIES[family], widths, titles, plots_dir,
                args.windows, args.window_labels, args.save_type,
                pre=args.pre, post=args.post,
            )
        return

    window = args.window if args.mode == "fold" else args.sample_type
    df = df_all.filter(pl.col("sample_type") == window)
    if df.height == 0:
        sys.exit(f"No rows with sample_type == {window}")

    for family in families:
        if args.mode == "fold":
            plot_fold_single(
                df, family, FAMILIES[family], widths, titles, plots_dir,
                window, args.save_type, pre=args.pre, post=args.post,
            )
        else:
            plot_family(
                df, family, FAMILIES[family], widths, titles, plots_dir,
                args.sample_type, args.save_type,
            )


if __name__ == "__main__":
    main()
