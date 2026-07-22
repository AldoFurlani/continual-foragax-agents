import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import tol_colors as tc
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def save(save_path: str, plot_name: str, save_type: str, f: Figure, **kwargs):
    """Save matplotlib figure to file."""
    save_dir = Path(save_path)

    file_path = save_dir / f"{plot_name}.{save_type}"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    f.savefig(file_path, bbox_inches="tight", pad_inches=0, **kwargs)
    print(f"Plot saved to: {file_path}")


sys.path.append(os.getcwd())

# ---------------------------------
# Matplotlib and Seaborn Settings
# ---------------------------------
sns.set_palette(tc.tol_cset("vibrant"))

# Set font
plt.rcParams.update(
    {
        "text.usetex": False,  # Don't use LaTeX
        "font.family": "serif",  # Use a serif font
        "font.serif": ["Linux Libertine O"],  # Specifically, use Linux Libertine
        "font.sans-serif": ["Linux Biolinum O"],  # Use Linux Biolinum for sans-serif
        "mathtext.fontset": "custom",  # Use custom math fonts
        "mathtext.rm": "Linux Libertine O",  # Roman text in math
        "mathtext.it": "Linux Libertine O:italic",  # Italic text in math
        "mathtext.bf": "Linux Libertine O:bold",  # Bold text in math
        "axes.unicode_minus": False,  # Ensure minus signs are rendered correctly
    }
)

# Set font sizes for better readability in papers
FONTSIZE = 16
plt.rcParams["axes.labelsize"] = FONTSIZE  # Axis labels
plt.rcParams["xtick.labelsize"] = FONTSIZE  # X-tick labels
plt.rcParams["ytick.labelsize"] = FONTSIZE  # Y-tick labels

# ---------------------
# Constants
# ---------------------
LABEL_MAP: Dict[str, str] = {
    "PPO": "PPO",
    "PPO_128": "PPO",
    "PPO_LN_128": "PPO",
    "PPO_LN_RT_128": "PPO (Simple Memory)",
    "PPO_LN_HINT_128": "PPO (CA)",
    "PPO_L2": "PPO (L2)",
    "ActorCriticMLP": "PPO",
    "ActorCriticMLPReLU": "PPO (ReLU)",
    "ActorCriticMLP-l2": "PPO (L2)",
    "ActorCriticMLP-l2-init": "PPO (L2 Init)",
    "ActorCriticMLP-world": "PPO (World)",
    "ActorCriticMLP-reward-trace": "PPO Simple Memory",
    "ActorCriticMLP-reset": "PPO (Head Reset)",
    "ActorCriticMLP-shrink-and-perturb": "PPO (S&P)",
    "RealTimeActorCriticMLP": "RTU-PPO",
    "RealTimeActorCriticMLPReLU": "RTU-PPO (ReLU)",
    "RealTimeActorCriticMLPMulti": "RTU-PPO (Multi)",
    "RealTimeActorCriticMLP-l2": "RTU-PPO (L2)",
    "PPO-RTU": "RTU-PPO",
    "PPO-RTU_128": "RTU-PPO",
    "PPO-RTU_L2": "RTU-PPO (L2)",
    "PPO-RTU_LN_128": "RTU-PPO (LN)",
    "PPO-RTU_LN_128_512": "RTU-PPO (LN, H512)",
    "RealTimeActorCriticMLP-world": "RTU-PPO (World)",
    "DQN": "DQN",
    "DQN_CReLU": "DQN (CReLU)",
    "DQN_L2": "DQN (L2)",
    "DQN_L2_Init": "DQN (L2 Init)",
    "DQN_LN": "DQN (LN)",
    "DQN_reward_trace": "DQN Simple Memory",
    "DQN_Reset_Head": "DQN (Head Reset)",
    "DQN_Hare_and_Tortoise": "DQN (Hare & Tortoise)",
    "DQN_Shrink_and_Perturb": "DQN (S&P)",
    "DQN_privileged": "DQN (Privileged)",
    "DQN_world": "DQN (World)",
    "DQN_LN_RT": "DQN (LN, RT)",
    "DRQN": "DRQN",
    "DRQN_LN_1_1": "DRQN (LN, 1-1)",
    "DRQN_1_1": "DRQN (1-1)",
    "DRQN_LN_0_2": "DRQN (LN, 0-2)",
    "DRQN_0_2": "DRQN (0-2)",
    "PT_DQN": "PT DQN",
    "Search-Brown": "Search (Brown)",
    "Search-Brown-Avoid-Green": "Search (+B-G)",
    "Search-Morel": "Search (Morel)",
    "Search-Morel-Avoid-Green": "Search (Morel, Avoid Green)",
    "Search-Nearest": "Search (Nearest)",
    "Search-Oracle": "Search (Oracle)",
    "Search-Oyster": "Search (Oyster)",
    "Search-9": "Search",
}

frozen_label_map = {}
for key in list(LABEL_MAP.keys()):
    frozen_label_map[f"{key}_greedy_frozen_5M"] = (
        f"{LABEL_MAP[key]} (5M)"
    )
    frozen_label_map[f"{key}_greedy_frozen_1M"] = (
        f"{LABEL_MAP[key]} (1M)"
    )
    frozen_label_map[f"{key}_frozen_1M"] = f"{LABEL_MAP[key]} (1M)"
    frozen_label_map[f"{key}_frozen_5M"] = f"{LABEL_MAP[key]} (F)"
LABEL_MAP.update(frozen_label_map)


YLABEL_MAP: Dict[str, str] = {
    "Ewm Reward": "Average Reward",
    "ewm_reward_5": "Average Reward",
    "ewm_reward_9": "Average Reward",
}


# Color scheme for plotting
# `Any`: tol_colors' tol_cset/tol_cmap return types don't expose the named color
# attributes / call signature to the type checker.
colorset: Any = tc.tol_cset("high_contrast")
sunset_colormap: Any = tc.tol_cmap("sunset")

# Biome colors for plotting
TWO_BIOME_COLORS: Dict[str, Any] = {
    "Morel": colorset.red,
    "Oyster": colorset.yellow,
    "Neither": colorset.blue,
}

WEATHER_BIOME_COLORS: Dict[str, Any] = {
    "Cold": sunset_colormap(0.0),
    "Neither": sunset_colormap(0.5),
    "Hot": sunset_colormap(1.0),
}

# Algorithm color mapping (vibrant palette)
_vibrant = tc.colorsets["vibrant"]
_muted = tc.colorsets["muted"]
COLOR_MAP: Dict[str, Any] = {
    # Raw alg keys
    "PPO-RTU_LN_HINT_128:9": _muted.rose,
    "DQN_LN_HINT:9": _muted.olive,
    "PPO_LN_RT_128:9": _vibrant.cyan,
    "PPO-RTU_LN_128:9": _vibrant.magenta,
    "Search-9": _vibrant.orange,
    "Search-Oracle": _vibrant.red,
    "PPO_LN_HINT_128:9": _muted.purple,
    "ActorCriticMLP": _muted.indigo,
    "ActorCriticMLP_frozen_5M": _muted.indigo,
    "DQN": _muted.green,
    "DQN_LN": _muted.green,
    "DQN_frozen_5M": _muted.green,
    "DRQN": "black",
    "DRQN_frozen_5M": "black",
    "RealTimeActorCriticMLP": _vibrant.magenta,
    "RealTimeActorCriticMLP_frozen_5M": _vibrant.magenta,
    "DQN_reward_trace": _vibrant.teal,
    "DQN_reward_trace_frozen_5M": _vibrant.teal,
    "ActorCriticMLP-reward-trace": _vibrant.cyan,
    "ActorCriticMLP-reward-trace_frozen_5M": _vibrant.cyan,
    # Mapped label keys
    "RTU-PPO (CA)": _muted.rose,
    "DQN (CA)": _muted.olive,
    "PPO (Simple Memory)": _vibrant.cyan,
    "RTU-PPO": _vibrant.magenta,
    "RTU-PPO (Frozen @ 5 M)": _vibrant.magenta,
    "Search": _vibrant.orange,
    "Search (Oracle)": _vibrant.red,
    "PPO (CA)": _muted.purple,
    "PPO": _muted.indigo,
    "PPO (Frozen @ 5 M)": _muted.indigo,
    "DQN (Frozen @ 5 M)": _muted.green,
    "DRQN (Frozen @ 5 M)": "black",
    "DQN (RT)": _vibrant.teal,
    "DQN (RT) (Frozen @ 5 M)": _vibrant.teal,
    "PPO (RT)": _vibrant.cyan,
    "PPO (RT) (Frozen @ 5 M)": _vibrant.cyan,
}


# ---------------------
# Mappings
# ---------------------
def get_biome_mapping(env: str) -> Dict[int, str]:
    if "TwoBiome" in env:
        return {-1: "Neither", 0: "Morel", 1: "Oyster"}
    if "Weather" in env:
        return {-1: "Neither", 0: "Hot", 1: "Cold"}
    raise ValueError(f"Unknown biome mapping for environment: {env}")


def get_object_mapping(env: str) -> Dict[int, str]:
    if "Foragax" in env:
        return {0: "Morel", 1: "Oyster", 2: "Chantrelle"}
    raise ValueError(f"Unknown object mapping for environment: {env}")


def get_ylabel_mapping(env: str) -> Dict[str, str]:
    """Get environment-specific ylabel mappings for metrics."""
    base_mapping = {
        "Rolling Reward 1000000": "Average Reward (1 M)",
        "Ewm Reward": "Average Reward",
        "Ewm Reward 5": "Average Reward",
        "Ewm Reward 9": "Average Reward",
        "Temperature": "Temperature",
    }

    if "Weather" in env:
        weather_mapping = {
            "Object Trace 1 2": r"Hot Trace ($10^{-2}$)",
            "Object Trace 2 2": r"Cold Trace ($10^{-2}$)",
            "Object Trace 1 1": r"Hot Trace ($10^{-1}$)",
            "Object Trace 2 1": r"Cold Trace ($10^{-1}$)",
        }
        base_mapping.update(weather_mapping)
    elif "TwoBiome" in env:
        twobiome_mapping = {
            "Object Trace 1 2": r"Morel Trace ($10^{-2}$)",
            "Object Trace 2 2": r"Oyster Trace ($10^{-2}$)",
            "Object Trace 3 2": r"Deathcap Trace ($10^{-1}$)",
            "Object Trace 4 2": r"Fake Trace ($10^{-1}$)",
            "Object Trace 1 1": r"Morel Trace ($10^{-1}$)",
            "Object Trace 2 1": r"Oyster Trace ($10^{-1}$)",
            "Object Trace 3 1": r"Deathcap Trace ($10^{-1}$)",
            "Object Trace 4 1": r"Fake Trace ($10^{-1}$)",
        }
        base_mapping.update(twobiome_mapping)

    return base_mapping


# ---------------------
# Data Loading
# ---------------------
def load_data(experiment_path: Path) -> pl.DataFrame:
    """Loads and preprocesses data from the specified experiment_path."""
    data_path = (
        Path("results")
        / experiment_path.relative_to(Path("experiments").resolve())
        / "data.parquet"
    )
    df = pl.read_parquet(data_path)
    df = df.with_columns(
        pl.col("alg").str.replace(r"_frozen_.*", "").alias("alg_base"),
        pl.when(pl.col("alg").str.contains("_frozen"))
        .then(pl.col("alg").str.extract(r"_frozen_(.*)", 1))
        .otherwise(None)
        .alias("freeze_steps_str"),
    )
    return df


# ---------------------
# Data Filtering
# ---------------------
def filter_by_alg_aperture(
    df: pl.DataFrame, filter_strings: Optional[List[str]]
) -> pl.DataFrame:
    if not filter_strings:
        return df

    conditions = []
    for faa in filter_strings:
        parts = faa.split(":")
        alg = parts[0]

        cond = pl.col("alg") == alg
        if len(parts) > 1:
            aperture = int(parts[1])
            cond = cond & (pl.col("aperture") == aperture)

        conditions.append(cond)

    return df.filter(pl.any_horizontal(conditions))


# ---------------------
# Argument Parsing
# ---------------------
class PlottingArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("path", type=str, help="Path to the experiment directory")
        self.add_argument(
            "--save-type", type=str, default="pdf", help="File format for saving plots"
        )
        self.add_argument(
            "--font-size", type=int, default=FONTSIZE, help="Font size for the plot."
        )
        self.add_argument(
            "--plot-name", type=str, default=None, help="Custom plot name"
        )
        self.add_argument(
            "--filter-algs",
            type=str,
            nargs="*",
            help="Algorithms to include in the plot",
        )
        self.add_argument(
            "--filter-seeds", type=int, nargs="*", help="Seeds to include in the plot"
        )
        self.add_argument(
            "--filter-alg-apertures",
            nargs="*",
            help="Filter for specific algorithm-aperture pairs (e.g., 'DQN:9').",
        )


def parse_plotting_args(parser: PlottingArgumentParser) -> argparse.Namespace:
    """Parses and returns common plotting arguments."""
    args = parser.parse_args()
    args.experiment_path = Path(args.path).resolve()

    # Update matplotlib font sizes with chosen font size
    if hasattr(args, "font_size"):
        plt.rcParams.update({
            "axes.labelsize": args.font_size,
            "xtick.labelsize": args.font_size,
            "ytick.labelsize": args.font_size,
            "font.size": args.font_size,
            "legend.fontsize": args.font_size,
        })

    return args


# ---------------------
# Plot Saving
# ---------------------
def save_plot(
    fig: Figure,
    experiment_path: Path,
    plot_name: str,
    save_type: str,
    **kwargs: Any,
):
    """Saves a matplotlib figure to the specified path."""
    save_path = experiment_path / "plots"
    save(
        save_path=str(save_path),
        plot_name=plot_name,
        save_type=save_type,
        f=fig,
        **kwargs,
    )


# ---------------------
# Plotting Utilities
# ---------------------
def format_metric_name(metric: str) -> str:
    """Formats a metric name for display."""
    import re
    if re.match(r"ewm_reward", metric):
        return YLABEL_MAP.get(metric, "Average Reward")
    metric = YLABEL_MAP.get(metric, metric)
    return metric.replace("_", " ").title()


def get_mapped_label(label: str, label_map: Optional[Dict[str, str]] = None, disable_fov: bool = False) -> str:
    """Get the mapped label, handling apertures and frozen variants."""
    if label_map and label in label_map:
        return label_map[label]
    if ":" in label:
        alg, aperture = label.split(":", 1)
        base_label = label_map.get(alg, alg) if label_map else alg
        if "frozen" in base_label.lower():
            # For frozen variants, put FOV inline with DQN and Frozen on new line
            dqn_part = base_label.split(" (")[0]
            frozen_part = base_label.split(" (", 1)[1]
            if disable_fov:
                return f"{dqn_part}\n({frozen_part}"
            else:
                return f"{dqn_part} (FOV {aperture})\n({frozen_part}"
        else:
            if disable_fov:
                return base_label
            return f"{base_label} (FOV {aperture})"
    return label


def get_legend_elements(labels: List[str], colors: Dict[str, Any]) -> List[Line2D]:
    """Creates legend elements for a plot."""
    return [
        Line2D([0], [0], color=colors[label], lw=2, label=LABEL_MAP.get(label, label))
        for label in labels
    ]


def despine(ax: Axes):
    """Removes top and right spines from a plot."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
