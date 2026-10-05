import os
import sys
import tol_colors as tc
import polars as pl
# sys.path.append(os.getcwd() + "/src")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib.pyplot as plt
import numpy as np
import re
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from rlevaluation.config import data_definition
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import (
    curve_percentile_bootstrap_ci,
    extract_learning_curves,
)

from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection

setDefaultConference("jmlr")
setFonts(20)

colorset = tc.colorsets["muted"]

PALETTE = [
    colorset.rose,
    colorset.indigo,
    colorset.teal,
    colorset.olive,
    colorset.purple,
    colorset.wine,
    colorset.green,
    # colorset.sand,
    colorset.cyan,
]

# Linestyles to distinguish families
LINESTYLES = {
    "RealTimeActorCriticMLP": "-",
    "ActorCriticMLP":        "-", 
    "Random": ":",                
}

# LINESTYLES = {
#     "RealTimeActorCriticConv-3": "-",
#     "RealTimeActorCriticConvEmb-3": "--",
#     "RealTimeActorCriticConvEmbNE-3": "-.",
#     "RealTimeActorCriticConvNE-3": ":",
#     "RealTimeActorCriticConvPooling-3": (0, (3, 1, 1, 1)),
#     "RealTimeActorCriticConvPoolingNE-3": (0, (5, 1)),
#     "RealTimeActorCriticMLP-3": (0, (1, 2)),
#     "RealTimeActorCriticMLPNE-3": (0, (3, 5, 1, 5)),
#     "Random": (0, (1, 1)),
# }

SINGLE = {
    "Random",
    "Search-Oracle",
    "Search-Nearest",
    "Search-Brown-Avoid-Green"
}

# Helper: strip optional "1M" token (with or without leading separator) so
# color is shared between 1M and non-1M variants

def base_without_1m(name: str) -> str:
    return re.sub(r"[-_]?1M", "", name)

if __name__ == "__main__":
    results = ResultCollection(Model=ExperimentModel, metrics=["ewm_reward"])
    dd = data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )
    env = Path(__file__).resolve().parent.name
    baselines_results = []

    # Collect sub-results grouped by FOV (aperture) so we can build a grid
    by_aperture = {}
    for key, sub_results in sorted(
        results.groupby_directory(level=4), key=lambda x: x[0]
    ):
        # level=3 directory is either an aperture like "9"/"15" or the literal "baselines"
        if str(key).lower() == "baselines":
            baselines_results = sub_results
            continue
        aperture = int(str(key))
        by_aperture.setdefault((env, aperture), []).extend(sub_results)

    # Determine unique apertures for this env (assumes a single env; if multiple, we create per-env figures)
    # Group by env first
    env_to_apertures = {}
    for (env, aperture), subs in by_aperture.items():
        env_to_apertures.setdefault(env, []).append(aperture)


    # --- Build a 2x3 figure for this env with specific comparisons ---
    fig, axes = plt.subplots(1, 1, squeeze=False, sharey=True, figsize=(15, 8))

    # Convenience mapping from (env, aperture) to list of results already exists: by_aperture
    # Helpers to identify algorithm families and variants
    def is_rtu(name: str) -> bool:
        return name.startswith("RealTimeActorCriticMLP")

    def is_ppo(name: str) -> bool:
        return name.startswith("ActorCriticMLP")

    def is_esm(name: str) -> bool:
        return name.startswith("ESM")

    def variant_token(name: str) -> str:
        # Return '', '1M', or '5M' based on filename tokens
        if "5M" in name:
            return "5M"
        if "1M" in name:
            return "1M"
        return ""

    def pretty_label(raw_alg: str, aperture: int | None, variant: str) -> str:
        # Map internal names to pretty labels requested in the prompt
        if raw_alg.startswith("Search-"):
            return raw_alg
        fam = "RTU-PPO" if is_rtu(raw_alg) else ("PPO" if is_ppo(raw_alg) else raw_alg)
        fov = f"FOV{aperture}" if aperture is not None else ""
        if variant == "1M":
            suffix = " frozen 1m"
        elif variant == "5M":
            suffix = " frozen 5m"
        else:
            suffix = ""
        if "l2" in raw_alg:
            suffix = " L2 " + suffix
        if "sinusoidal" in raw_alg:
            suffix = " sinusoidal " + suffix
        if "reward-trace" in raw_alg:
            suffix = " reward-trace " + suffix
        if "transfer" in raw_alg:
            suffix = " transfer " + suffix
        return f"{fam} {fov}{suffix}".strip()

    # Line styles by variant: base solid, 1M dashed, 5M dotted
    def linestyle_for_variant(variant: str) -> str:
        return "-" if variant == "" else ("--" if variant == "1M" else ":")

    # Build color assignments so that base and its frozen variants share a color per family+FOV
    PALETTE_CYCLE = PALETTE  # reuse global palette
    color_index = 0
    base_color_map = {}

    def color_for(base_key: str) -> str:
        global color_index
        if base_key not in base_color_map:
            base_color_map[base_key] = PALETTE_CYCLE[color_index % len(PALETTE_CYCLE)]
            color_index += 1
        return base_color_map[base_key]

    oracle_store = {"xs": None, "ys": None}
    
    # Utility to find and plot a single selection
    def plot_selection(ax, family: str, aperture: int | None, desired_variant: str | None, require_l2: bool=False, require_reward_trace: bool=False, require_sinusoidal: bool=False, color_index:int=0, plot_all_lines:bool=False, require_transfer: bool=False):
        # family in {"SINGLE:Search-Oracle", "SINGLE:Search-Nearest", "RTU", "PPO"}
        if family.startswith("SINGLE:"):
            single_name = family.split(":", 1)[1]
            # Search SINGLE results across baselines_results; these are aperture-agnostic
            found = None
            for ar in baselines_results:
                if ar.filename == single_name:
                    found = ar
                    break
            if not found:
                return
            ar = found
            print(ar.filename)
            df = ar.load(end=10_000_000)
            if df is None:
                return
            cols = set(dd.hyper_cols).intersection(df.columns)
            hyper_vals = {col: df[col][0] for col in cols}
            seeds = sorted(df['seed'].unique().to_list())
            ys, xs, = [], []
            for seed in seeds:
                seed_df = df.filter(pl.col('seed') == seed)
                x, y = extract_learning_curves(seed_df, hyper_vals=hyper_vals, metric="ewm_reward")
                x = np.asarray(x)
                y = np.asarray(y)
                xs.append(x)
                ys.append(y)
            # xs, ys = extract_learning_curves(df, hyper_vals=hyper_vals, metric="ewm_reward")
            # xs = np.asarray(xs)
            # ys = np.asarray(ys)
            xs = np.vstack(xs)
            ys = np.vstack(ys)
            mask = xs[0] > 1000
            xs = xs[:, mask]
            ys = ys[:, mask]
            if ar.filename == "Search-Oracle":
                oracle_store["xs"], oracle_store["ys"] = xs, ys
            # ys = ys / oracle_store["ys"]
            assert np.all(np.isclose(xs[0], xs))
            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0), y=ys, statistic=Statistic.mean, iterations=10000
            )
            base_key = f"SINGLE-{single_name}"
            variant_color = PALETTE_CYCLE[(color_index) % len(PALETTE_CYCLE)]
            label = pretty_label(single_name, None, "")
            ax.plot(xs[0], res.sample_stat, label=label, color=variant_color, linewidth=1.0, alpha=0.6)
            if len(ys) >= 5 and not plot_all_lines:
                ax.fill_between(xs[0], res.ci[0], res.ci[1], color=variant_color, alpha=0.1)
            else:
                for y in ys:
                    ax.plot(xs[0], y, color=variant_color, linewidth=0.2)
            return

        # Family selections (RTU or PPO) for a specific aperture and variant
        sub_results = by_aperture.get((env, aperture), []) if aperture is not None else []
        # desired_variant in {None (any), "" (base), "1M", "5M"}
        for ar in sub_results:
            name = ar.filename
            # Exclude any agent with "world" in its name
            if "world" in name.lower():
                continue
            if require_reward_trace and 'reward-trace' not in name:
                continue
            if not require_reward_trace and 'reward-trace' in name:
                continue
            if require_l2 and '-l2' not in name:
                continue
            if not require_l2 and '-l2' in name:
                continue
            if require_sinusoidal and '-sinusoidal' not in name:
                continue
            if not require_sinusoidal and '-sinusoidal' in name:
                continue
            if require_transfer and '-transfer' not in name:
                continue
            if not require_transfer and '-transfer' in name:
                continue
            if family == "RTU" and not is_rtu(name):
                continue
            if family == "PPO" and not is_ppo(name):
                continue
            if family == "ESM" and not is_esm(name):
                continue
            if family == "DQN" and not name == "DQN":
                continue
            vt = variant_token(name)
            if desired_variant is None:
                pass  # accept any
            elif desired_variant == "":
                if vt != "":
                    continue
            else:
                if vt != desired_variant:
                    continue
            print(name)
            # We have a match; plot it and return
            df = ar.load(end=10_000_000)
            if df is None:
                return
            cols = set(dd.hyper_cols).intersection(df.columns)
            hyper_vals = {col: df[col][0] for col in cols}
            seeds = sorted(df['seed'].unique().to_list())
            ys, xs, = [], []
            for seed in seeds:
                seed_df = df.filter(pl.col('seed') == seed)
                x, y = extract_learning_curves(seed_df, hyper_vals=hyper_vals, metric="ewm_reward")
                x = np.asarray(x)
                y = np.asarray(y)
                xs.append(x)
                ys.append(y)
            # xs, ys = extract_learning_curves(df, hyper_vals=hyper_vals, metric="ewm_reward")
            # xs = np.asarray(xs)
            # ys = np.asarray(ys)
            xs = np.vstack(xs)
            ys = np.vstack(ys)
            mask = xs[0] > 1000
            xs = xs[:, mask]
            ys = ys[:, mask]
            # ys = ys / oracle_store["ys"]
            assert np.all(np.isclose(xs[0], xs))
            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0), y=ys, statistic=Statistic.mean, iterations=10000
            )
            base_key = f"{family}-FOV{aperture}"
            variant_color = PALETTE_CYCLE[(color_index) % len(PALETTE_CYCLE)]
            label = pretty_label(name, aperture, vt)
            ax.plot(xs[0], res.sample_stat, label=label, color=variant_color, linewidth=1.0)
            if len(ys) >= 5 and not plot_all_lines:
                ax.fill_between(xs[0], res.ci[0], res.ci[1], color=variant_color, alpha=0.1)
            else:
                for y in ys:
                    ax.plot(xs[0], y, color=variant_color, linewidth=0.2)
            return
        # If no match found, just return silently
        return

    # ---------------- Subplot 1 ----------------
    ax = axes[0][0]
    # Two search baselines
    # RTU/PPO at FOV 9 and 15 (base, non-frozen)
    plot_selection(ax, "SINGLE:Search-Oracle", None, "", color_index=-1)
    plot_selection(ax, "SINGLE:Search-Nearest", None, "", color_index=-2)
    # plot_selection(ax, "SINGLE:Search-Brown-Avoid-Green", None, "", color_index=-2)
    # plot_selection(ax, "RTU", 9, "5M", color_index=0)
    plot_selection(ax, "PPO", 9, "", color_index=1, require_reward_trace=True)
    plot_selection(ax, "PPO", 9, "", color_index=2)
    # plot_selection(ax, "RTU", 15, "5M", color_index=2)
    # plot_selection(ax, "PPO", 9, "5M", color_index=3, require_reward_trace=True)
    # plot_selection(ax, "PPO", 5, "5M", color_index=4, require_reward_trace=True)
    # plot_selection(ax, "PPO", 15, "5M", color_index=5, require_reward_trace=True)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Average Reward")
    ax.set_title(f"{env} — Baselines")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=1, loc="best", frameon=False, fontsize=12)
    

    # ax = axes[0][1]
    # # # Two search baselines
    # plot_selection(ax, "SINGLE:Search-Oracle", None, "", color_index=-1)
    # # plot_selection(ax, "SINGLE:Search-Nearest", None, "", color_index=-2)
    # # plot_selection(ax, "SINGLE:Search-Brown-Avoid-Green", None, "", color_index=-2)
    # # RTU/PPO at FOV 9 and 15 (base, non-frozen)
    # plot_selection(ax, "PPO", 5, "", color_index=0, require_reward_trace=True)
    # plot_selection(ax, "PPO", 9, "", color_index=1, require_reward_trace=True)
    # plot_selection(ax, "PPO", 15, "", color_index=2, require_reward_trace=True)
    # plot_selection(ax, "PPO", 5, "", color_index=3, require_l2=True)
    # plot_selection(ax, "PPO", 9, "", color_index=4, require_l2=True)
    # plot_selection(ax, "PPO", 15, "", color_index=5, require_l2=True)
    # # plot_selection(ax, "PPO", 15, "", color_index=2, require_reward_trace=True)
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    # ax.set_xlabel("Time steps")
    # ax.set_ylabel("Average Reward")
    # ax.set_title(f"{env} — Baselines Extra Info & FOV 9/15")

    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.legend(ncol=1, loc="best", frameon=False, fontsize=12)

    # # ---------------- Subplot 2 ----------------
    # ax = axes[0][2]
    # # RTU-PPO FOV9: base, frozen 1m, frozen 5m
    # plot_selection(ax, "RTU", 9, "", color_index=0)
    # plot_selection(ax, "RTU", 9, "1M", color_index=1)
    # plot_selection(ax, "RTU", 9, "5M", color_index=2)
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    # ax.set_xlabel("Time steps")
    # ax.set_title(f"{env} — RTU-PPO FOV 9: base vs frozen")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.legend(ncol=1, loc="best", frameon=False, fontsize=12)
    # ax.set_ylim(-1.5,1.5)

    # # ---------------- Subplot 3 ----------------
    # ax = axes[0][3]
    # # PPO FOV9: frozen 1m, frozen 5m
    # plot_selection(ax, "PPO", 9, "", color_index=0)
    # plot_selection(ax, "PPO", 9, "1M", color_index=1)
    # plot_selection(ax, "PPO", 9, "5M", color_index=2)
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    # ax.set_xlabel("Time steps")
    # ax.set_title(f"{env} — PPO FOV 9: frozen")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.legend(ncol=1, loc="best", frameon=False, fontsize=12)
    # ax.set_ylim(-1.5,1.5)

    # # ---------------- Subplot 4 ----------------
    # ax = axes[1][0]
    # # Two search baselines -l2
    # plot_selection(ax, "SINGLE:Search-Oracle", None, "")
    # plot_selection(ax, "SINGLE:Search-Nearest", None, "")
    # plot_selection(ax, "SINGLE:Search-Brown-Avoid-Green", None, "")
    # # RTU/PPO at FOV 9 and 15 (base, non-frozen) -l2
    # plot_selection(ax, "RTU", 9, "", require_l2=True)
    # plot_selection(ax, "PPO", 9, "", require_l2=True)
    # plot_selection(ax, "RTU", 15, "", require_l2=True)
    # plot_selection(ax, "PPO", 15, "", require_l2=True)
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    # ax.set_xlabel("Time steps")
    # ax.set_ylabel("Average Reward")
    # ax.set_title(f"{env} — Baselines-l2 & FOV 9/15")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.legend(ncol=1, loc="best", frameon=False, fontsize=12)

    # # ---------------- Subplot 5 ----------------
    # ax = axes[1][1]
    # # RTU-PPO FOV9: base-l2, frozen 1m-l2, frozen 5m-l2
    # plot_selection(ax, "RTU", 9, "", require_l2=True)
    # plot_selection(ax, "RTU", 9, "1M", require_l2=True)
    # plot_selection(ax, "RTU", 9, "5M", require_l2=True)
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    # ax.set_xlabel("Time steps")
    # ax.set_title(f"{env} — RTU-PPO FOV 9-l2: base vs frozen")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.legend(ncol=1, loc="best", frameon=False, fontsize=12)

    # # ---------------- Subplot 6 ----------------
    # ax = axes[1][2]
    # # PPO FOV9: frozen 1m-l2, frozen 5m-l2
    # plot_selection(ax, "PPO", 9, "", require_l2=True)
    # plot_selection(ax, "PPO", 9, "1M", require_l2=True)
    # plot_selection(ax, "PPO", 9, "5M", require_l2=True)
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    # ax.set_xlabel("Time steps")
    # ax.set_title(f"{env} — PPO FOV 9-l2: frozen")
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.legend(ncol=1, loc="best", frameon=False, fontsize=12)

    # Save one figure per env containing the 3 subplots
    path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
    save(
        save_path=f"{path}/plots",
        plot_name=f"three_panel",
        save_type="pdf",
        f=fig,
        width=2,
        height_ratio=1/2,
    )

    plt.close(fig)
