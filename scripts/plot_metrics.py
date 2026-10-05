"""Plot plasticity metrics (NTK rank/eff-rank, churn, weight norms) over
training, comparing one or more agents.

Like ``src/learning_curve.py``, color encodes the *agent*: each line is one
agent's across-seed mean with a shaded 95%% bootstrap CI band, so agents can be
compared directly and the band shows whether a trend is real or seed noise.
Agent colors come from the same shared ``COLOR_MAP`` / Paul Tol palette as
``learning_curve.py``.

The figure facets by (network role, metric): rows are the network trunks --
**actor** on top, **critic** below -- and columns are the metrics.  So a metric
that measures both trunks (e.g. weight drift) becomes an actor subplot above a
critic subplot rather than four cramped lines on one axis; the actor/critic
distinction is carried by the row (labelled once on the left), leaving color free
to mean only the agent.  All panels share the env-step x-axis; a metric's actor
and critic rows share a y-axis when their ranges are comparable (see ``share_y``
in ``METRICS``).  Pass a single ``--metric`` for one metric's actor+critic
column; ``--metrics a b c`` for several columns; neither to plot every metric
that has data.

Each friendly metric name resolves to whichever underlying column an agent
actually wrote: DQN runs store ``ntk_rank`` / ``churn_norm`` (a single head),
while PPO runs store separate ``value_*`` / ``policy_*`` columns.  So e.g.
``--metric ntk_rank`` plots ``policy_ntk_rank`` in the actor row and
``value_ntk_rank`` (or DQN's single ``ntk_rank``) in the critic row.

Examples:
    # every metric, actor row over critic row, one line per agent
    python scripts/plot_metrics.py \
        -e experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax/ForagaxSquareWaveTwoBiome-v11 \
        -a PPO_LN_128 PPO_LN_256 -f 9

    # a single metric -> its actor and critic subplots
    python scripts/plot_metrics.py -e <exp> -a DQN PPO_LN_128 -f 9 \
        --metric churn --plot-name dqn_vs_ppo_churn

    # list the metric names you can pass to --metric / --metrics
    python scripts/plot_metrics.py --list-metrics
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # Go up from scripts/ to repo root
sys.path.insert(0, str(ROOT / "src"))

try:  # optional: prettier legend labels + shared colors, but don't hard-depend
    from plotting_utils import COLOR_MAP, LABEL_MAP, get_mapped_label
except Exception:  # pragma: no cover - fallback when src isn't importable
    COLOR_MAP = {}
    LABEL_MAP = {}

    def get_mapped_label(label, label_map=None, disable_fov=False):
        return label


# Paul Tol qualitative palette, mirroring learning_curve.py's coloring: vibrant
# first, then any muted colors not already in vibrant, then (only when there are
# more agents than that) a generated husl palette.
try:
    import tol_colors as tc

    _VIBRANT = list(tc.colorsets["vibrant"])
    _MUTED = list(tc.colorsets["muted"])
    _COMBINED = _VIBRANT + [c for c in _MUTED if c not in _VIBRANT]
except Exception:  # pragma: no cover
    _VIBRANT = _COMBINED = []


def build_agent_colors(agent_names: list) -> dict:
    """One color per agent, consistent across panels, matching learning_curve.py.

    Prefers the shared ``COLOR_MAP`` entry for an agent (so an agent keeps the
    same color across every plot in the repo); otherwise draws from the Paul Tol
    vibrant palette, widening to vibrant+muted and finally a husl palette when
    there are more agents than the base palette holds.  ``COLOR_MAP`` agents do
    not consume a fallback-cycle slot (same accounting as learning_curve.py).
    """
    n = len(agent_names)
    if _VIBRANT and n <= len(_VIBRANT):
        cycle = _VIBRANT
    elif _COMBINED and n <= len(_COMBINED):
        cycle = _COMBINED
    elif _COMBINED:
        try:
            import seaborn as sns

            cycle = sns.color_palette("husl", n)
        except Exception:  # pragma: no cover
            cycle = _COMBINED
    else:  # tol_colors unavailable -> matplotlib default cycle
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", []) or ["C0"]

    colors, fallback_idx = {}, 0
    for agent in agent_names:
        if agent in COLOR_MAP:
            colors[agent] = COLOR_MAP[agent]
        else:
            colors[agent] = cycle[fallback_idx % len(cycle)]
            fallback_idx += 1
    return colors


# Friendly metric name -> panel spec.  A metric holds one or more `series`; each
# series has a `role` (the network trunk it measures: "actor" / "critic") and a
# `cols` *priority list* of underlying parquet/npz column names.  For each agent
# the first column that has finite data is used, so one friendly name works
# across agent types (DQN's single head vs the actor/critic heads).
#
# The plot facets by (metric, role): rows are the network roles (actor on top,
# critic below), columns are the metrics.  Color encodes *only the agent*, and
# the actor/critic distinction is carried by which row a subplot is in -- so a
# metric that used to cram four lines onto one axis is now two uncluttered
# subplots of one line per agent.  A per-series `label` overrides the metric
# `label` for that row's y-axis (used by churn, whose actor and critic are
# different quantities).  `share_y` links the y-limits of a metric's actor and
# critic rows; set it False when their ranges differ enough that a shared axis
# would crush one (e.g. NTK effective rank: actor ~8-10 vs critic ~2-4).  `log`
# uses a log y-axis.
METRICS = {
    "ntk_rank": {
        "label": "NTK Rank",
        "title": "NTK Rank",
        "log": False,
        "share_y": True,
        "series": [
            {"role": "actor", "cols": ["policy_ntk_rank"]},
            {"role": "critic", "cols": ["value_ntk_rank", "ntk_rank"]},
        ],
    },
    "ntk_eff_rank": {
        "label": "NTK Effective Rank",
        "title": "NTK Effective Rank",
        "log": False,
        "share_y": False,  # actor ~8-10, critic ~2-4: shared axis crushes critic
        "series": [
            {"role": "actor", "cols": ["policy_ntk_eff_rank"]},
            {"role": "critic", "cols": ["value_ntk_eff_rank"]},
        ],
    },
    # Churn: actor is a KL divergence, critic a relative (scale-invariant) MSE --
    # different quantities, so independent y-axes and per-row labels.
    "churn": {
        "label": "Churn",
        "title": "Churn",
        "log": False,
        "share_y": False,
        "series": [
            {
                "role": "actor",
                "label": "Actor Churn",
                "cols": ["policy_churn"],
            },
            {
                "role": "critic",
                "label": "Critic Churn",
                "cols": ["value_churn", "churn_norm"],
            },
        ],
    },
    "weight_drift": {
        "label": "Weight Drift",
        "title": "Weight Drift",
        "log": False,
        "share_y": True,
        "series": [
            {"role": "actor", "cols": ["weight_drift_pi"]},
            {"role": "critic", "cols": ["weight_drift_vf"]},
        ],
    },
    "weight_update_norm": {
        "label": "Weight Update Norm",
        "title": "Weight Update Norm",
        "log": False,
        "share_y": True,
        "series": [
            {"role": "actor", "cols": ["weight_update_norm_pi"]},
            {"role": "critic", "cols": ["weight_update_norm_vf"]},
        ],
    },
    "weight_norm": {
        "label": "Weight Norm",
        "title": "Weight Norm",
        "log": False,
        "share_y": True,
        "series": [
            {"role": "actor", "cols": ["weight_norm_pi"]},
            {"role": "critic", "cols": ["weight_norm_vf"]},
        ],
    },
}

# Canonical top-to-bottom row order for the network roles.
ROLE_ORDER = ["actor", "critic", "total"]


def _normalize_exp(exp: str) -> str:
    """Strip a leading experiments/ or results/ prefix and trailing slash."""
    exp = exp.strip().rstrip("/")
    for prefix in ("experiments/", "results/"):
        if exp.startswith(prefix):
            exp = exp[len(prefix) :]
    return exp


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-e",
        "--exp",
        help="Experiment path, e.g. experiments/E136-big/foragax/ForagaxBig-v5 "
        "(a leading experiments/ or results/ prefix is optional)",
    )
    p.add_argument(
        "-a",
        "--agent",
        nargs="+",
        help="One or more agent / algorithm names (e.g. DQN PPO_LN_128). Each "
        "is drawn as its own line in every metric panel.",
    )
    p.add_argument(
        "-f",
        "--fov",
        nargs="+",
        type=int,
        help="Field-of-view / aperture size. Pass one value (applied to every "
        "agent) or one per agent.",
    )
    p.add_argument(
        "--metric",
        help="Single metric to plot -> a single graph. See --list-metrics.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        help="Metrics to plot as a grid of panels. See --list-metrics. "
        "If neither --metric nor --metrics is given, every metric with data "
        "is plotted.",
    )
    p.add_argument(
        "--plot-name",
        help="Output filename stem (without extension). Defaults to a name "
        "derived from the metric(s) and agent(s).",
    )
    p.add_argument(
        "--ci",
        type=float,
        default=95.0,
        help="Confidence level (%%) for the across-seed bootstrap band (default 95).",
    )
    p.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Bootstrap resamples for the CI band (default 1000, matching seaborn).",
    )
    p.add_argument("--save-type", default="png", help="Image format (png, pdf, ...).")
    p.add_argument(
        "--no-legend", action="store_true", help="Suppress the per-agent legend."
    )
    p.add_argument(
        "--split-roles",
        action="store_true",
        help="Save each network role (actor, critic, ...) as its own figure/file "
        "instead of stacking them as rows in one image.",
    )
    p.add_argument(
        "--list-metrics",
        action="store_true",
        help="Print the available metric names and exit.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Write the plot to the repo root instead of the experiment's "
        "metrics/ folder (handy for quick one-off checks).",
    )
    return p.parse_args()


def load_runs(data_path: Path) -> list:
    """All per-run npz dicts under `data_path` (one file per seed).

    Read the raw npz (not data.parquet) to keep metrics at native resolution:
    DQN writes one value per env step (NaN except every ntk_freq), PPO writes one
    value per *update*.  The parquet reader repeats PPO per-update arrays to
    per-step length, which makes constant series indistinguishable from a single
    repeated point.
    """
    runs = []
    for f in sorted(Path(data_path).glob("*.npz")):
        with np.load(f) as d:
            runs.append({k: np.asarray(d[k]) for k in d.keys()})
    return runs


def metric_matrix(runs: list, col: str):
    """Per-seed aligned (x_steps, Y) for `col`, or None if absent/empty.

    A metric array has one entry per *measurement* (per env step for DQN, per
    update for PPO).  Its x-axis is recovered in env steps by comparing its
    length to the per-step `rewards` array: `steps_per_point = len(rewards) /
    len(metric)` (1 for DQN, rollout_steps for PPO).

    Unlike the old seed-averaging path, seeds are kept as separate rows so the
    caller can bootstrap a confidence interval across them.  Runs are truncated
    to the shortest common length (the metric schedule is identical across
    seeds, so column ``j`` is the same measurement step for every seed) and
    returned as ``Y`` of shape ``[n_seeds, n_points]`` -- still NaN on
    non-measurement steps; the caller drops those columns after aggregating.
    """
    arrays = []
    steps_per_point = 1.0
    for r in runs:
        if col not in r:
            continue
        v = np.asarray(r[col], dtype=float).reshape(-1)
        if v.shape[0] == 0:
            continue
        base_len = r["rewards"].reshape(-1).shape[0] if "rewards" in r else v.shape[0]
        steps_per_point = base_len / v.shape[0]
        arrays.append(v)

    if not arrays:
        return None

    m = min(a.shape[0] for a in arrays)
    Y = np.stack([a[:m] for a in arrays], axis=0)  # [n_seeds, m]
    x = np.arange(m) * steps_per_point
    return x, Y


def _bootstrap_ci(Y: np.ndarray, n_boot: int, ci: float, seed: int = 0):
    """Across-seed mean and a percentile bootstrap CI band, per measurement step.

    ``Y`` is ``[n_seeds, n_points]``.  Returns ``(mean, lo, hi)`` each
    ``[n_points]``: ``mean`` is the across-seed ``nanmean``; ``lo``/``hi`` are
    the ``ci``% percentile-bootstrap interval of that mean, resampling seeds
    with replacement ``n_boot`` times -- the same nonparametric bootstrap
    seaborn's ``errorbar=("ci", 95)`` uses in learning_curve.py.  With fewer than
    two seeds the CI is undefined, so ``lo``/``hi`` are NaN and the band is
    simply not drawn.  A fixed ``seed`` keeps the band reproducible run-to-run.
    """
    n_seeds = Y.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
        mean = np.nanmean(Y, axis=0)
        if n_seeds < 2:
            nan = np.full_like(mean, np.nan)
            return mean, nan, nan
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
        boot = np.nanmean(Y[idx], axis=1)  # [n_boot, n_points]
        half = (100.0 - ci) / 2.0
        lo = np.nanpercentile(boot, half, axis=0)
        hi = np.nanpercentile(boot, 100.0 - half, axis=0)
    return mean, lo, hi


def resolve_series(runs: list, cols: list, n_boot: int = 1000, ci: float = 95.0):
    """First ``(col, (x, mean, lo, hi))`` in the priority list `cols` with finite
    data for this agent, or None.

    Seeds are aggregated into an across-seed mean and a `ci`% bootstrap band,
    with the non-measurement (all-NaN) steps dropped after aggregation.  The
    bootstrap runs only on the finite columns, so the sparse metric schedule
    keeps it cheap regardless of the per-step training length.
    """
    for col in cols:
        mat = metric_matrix(runs, col)
        if mat is None:
            continue
        x, Y = mat
        # Restrict to real measurement steps before bootstrapping (the metric is
        # NaN everywhere else), then aggregate.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            keep = np.isfinite(np.nanmean(Y, axis=0))
        if not keep.any():
            continue
        mean, lo, hi = _bootstrap_ci(Y[:, keep], n_boot, ci)
        return col, (x[keep], mean, lo, hi)
    return None


def _format_xaxis(ax):
    """Scale the x-axis to a readable power of ten (e.g. Time steps x10^6)."""
    x_max = ax.get_xlim()[1]
    if x_max <= 0:
        ax.set_xlabel("Step")
        return
    magnitude = 10 ** int(np.floor(np.log10(x_max)))
    scale = magnitude * 10 if x_max / magnitude > 50 else magnitude
    power = int(np.log10(scale))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / scale:.10g}"))
    ax.set_xlabel(f"Time steps ($\\times 10^{{{power}}}$)")


def render_figure(
    roles: list, metric_names: list, agents: list, agent_colors: dict, args
) -> "plt.Figure":
    """Build one figure: rows are `roles`, columns are `metric_names`.

    Factored out of `main` so it can be called once for the combined actor+critic
    figure, or once per role when `--split-roles` asks for separate files.
    """
    n_rows, n_cols = len(roles), len(metric_names)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        squeeze=False,
        sharex=True,  # every panel is the same env-step axis
        constrained_layout=True,  # reserves space for the legend/row labels
        # itself, instead of tight_layout's single guess before they're drawn
    )

    for c_idx, name in enumerate(metric_names):
        spec = METRICS[name]
        col_axes = []  # populated axes in this column, for optional y-sharing
        for r_idx, role in enumerate(roles):
            ax = axes[r_idx][c_idx]
            series = next((s for s in spec["series"] if s["role"] == role), None)
            if series is None:
                ax.axis("off")  # this metric has no such role (blank cell)
                continue

            n_plotted = 0
            for agent, fov, runs in agents:
                resolved = resolve_series(runs, series["cols"], args.n_boot, args.ci)
                if resolved is None:
                    continue
                _, (x, mean, lo, hi) = resolved
                color = agent_colors[agent]
                # Color == agent; the actor/critic distinction is the row, so no
                # per-series linestyle/marker is needed.
                ax.plot(x, mean, color=color, label=get_mapped_label(agent, LABEL_MAP))
                # Across-seed bootstrap CI band; skipped when <2 seeds (NaN lo/hi).
                if np.isfinite(lo).any():
                    ax.fill_between(x, lo, hi, color=color, alpha=0.2, linewidth=0)
                n_plotted += 1

            # y-label only when it says something the column title doesn't (e.g.
            # churn's per-row units); otherwise the title alone already names the
            # metric, and the role is labelled once per row further below.
            ylabel = series.get("label", spec["label"])
            if ylabel != spec["title"]:
                ax.set_ylabel(ylabel)
            if r_idx == 0:
                ax.set_title(spec["title"])
            if spec["log"] and n_plotted > 0:
                ax.set_yscale("log")
            ax.spines[["top", "right"]].set_visible(False)
            if n_plotted == 0:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                col_axes.append(ax)
            # x-axis label only on the bottom-most row.
            if r_idx == n_rows - 1:
                _format_xaxis(ax)

        # Optionally tie the actor/critic rows of this metric to one y-range so
        # their magnitudes stay directly comparable (skipped when their ranges
        # differ too much -- e.g. NTK effective rank -- via the registry flag).
        if spec.get("share_y") and len(col_axes) > 1:
            lo_y = min(a.get_ylim()[0] for a in col_axes)
            hi_y = max(a.get_ylim()[1] for a in col_axes)
            for a in col_axes:
                a.set_ylim(lo_y, hi_y)

    # Label each row (network role) once, to the left of the leftmost column.
    # Skipped when there's only one role (its own title already says which role
    # this figure is, e.g. when --split-roles saves it standalone).
    if n_rows > 1:
        for r_idx, role in enumerate(roles):
            axes[r_idx][0].annotate(
                role.capitalize(),
                xy=(0, 0.5),
                xytext=(-axes[r_idx][0].yaxis.labelpad - 18, 0),
                xycoords=axes[r_idx][0].yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=90,
                fontweight="bold",
            )

    # Single agent legend (color == agent), since roles are now rows not colors.
    if not args.no_legend and len(agents) > 1:
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0],
                [0],
                color=agent_colors[a],
                lw=2,
                label=get_mapped_label(a, LABEL_MAP),
            )
            for a, _, _ in agents
        ]
        fig.legend(
            handles=handles,
            frameon=False,
            loc="outside upper center",
            ncols=len(handles),
        )

    return fig


def main():
    args = parse_args()

    if args.list_metrics:
        print("Available metrics (pass to --metric / --metrics):\n")
        for name, spec in METRICS.items():
            parts = [f"{s['role']}: {'/'.join(s['cols'])}" for s in spec["series"]]
            print(f"  {name:<20} {spec['label']}  <- {'; '.join(parts)}")
        return

    if not (args.exp and args.agent and args.fov):
        sys.exit(
            "-e/--exp, -a/--agent and -f/--fov are required (unless --list-metrics)."
        )

    # Resolve which metrics to draw and validate names up front.
    if args.metric and args.metrics:
        sys.exit("Pass either --metric or --metrics, not both.")
    requested = [args.metric] if args.metric else args.metrics
    if requested:
        unknown = [m for m in requested if m not in METRICS]
        if unknown:
            sys.exit(
                f"Unknown metric(s): {unknown}. Run --list-metrics to see valid names."
            )

    # Broadcast a single fov to every agent, else require one fov per agent.
    if len(args.fov) == 1:
        fovs = args.fov * len(args.agent)
    elif len(args.fov) == len(args.agent):
        fovs = args.fov
    else:
        sys.exit(
            f"--fov must be one value or one per agent "
            f"({len(args.agent)} agents, got {len(args.fov)} fovs)."
        )

    exp = _normalize_exp(args.exp)
    metrics_dir = ROOT / "experiments" / exp / "metrics"

    # Load every agent's runs once.
    agents = []  # list of (agent, fov, runs)
    for agent, fov in zip(args.agent, fovs):
        data_path = ROOT / "results" / exp / str(fov) / agent / "data"
        runs = load_runs(data_path)
        print(f"{agent} (fov {fov}): loaded {len(runs)} run(s) from {data_path}")
        if not runs:
            print(f"  WARNING: no npz files found for {agent} — skipping.")
            continue
        agents.append((agent, fov, runs))

    if not agents:
        sys.exit("No data loaded for any agent.")

    # Decide the metric list.  Default: every metric that at least one agent has
    # finite data for, in registry order.
    if requested:
        metric_names = requested
    else:
        metric_names = [
            name
            for name, spec in METRICS.items()
            if any(
                resolve_series(runs, s["cols"])
                for _, _, runs in agents
                for s in spec["series"]
            )
        ]
        if not metric_names:
            sys.exit("None of the known metrics have finite data for these agents.")

    # One color per agent, consistent across panels (shared COLOR_MAP + Paul Tol
    # palette, matching learning_curve.py).
    agent_colors = build_agent_colors([agent for agent, _, _ in agents])

    # Facet by (role, metric): rows are network roles (actor on top, critic
    # below), columns are metrics.  A role row exists only if some selected
    # metric measures it; a (role, metric) cell with no matching series is blank.
    roles = [
        r
        for r in ROLE_ORDER
        if any(s["role"] == r for name in metric_names for s in METRICS[name]["series"])
    ]
    # Filename stem: prefer --plot-name, else derive from metric(s) + agent(s).
    agents_str = "_".join(agent for agent, _, _ in agents)
    if args.plot_name:
        base_stem = args.plot_name
    elif len(metric_names) == 1:
        base_stem = f"metrics_{metric_names[0]}_{agents_str}"
    else:
        base_stem = f"metrics_{agents_str}"

    out_dir = ROOT if args.test else metrics_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --split-roles: one figure/file per role (e.g. actor churn and critic churn
    # as separate images) instead of stacking every role as a row of one figure.
    role_groups = [[r] for r in roles] if args.split_roles else [roles]

    for role_group in role_groups:
        fig = render_figure(role_group, metric_names, agents, agent_colors, args)
        stem = base_stem if len(role_groups) == 1 else f"{base_stem}_{role_group[0]}"
        out = out_dir / f"{stem}.{args.save_type}"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
