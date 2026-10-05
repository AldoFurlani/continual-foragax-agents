import os
import sys
from pathlib import Path
# sys.path.append(os.getcwd() + '/src')
ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib.pyplot as plt
import numpy as np
from experiment.tools import parseCmdLineArgs
from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection


from PyExpPlotting.matplot import save, setDefaultConference
import rlevaluation.hypers as Hypers
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import TimeSummary, extract_learning_curves, curve_percentile_bootstrap_ci
from rlevaluation.config import data_definition
from rlevaluation.interpolation import compute_step_return
from utils.plotting import select_colors

setDefaultConference('jmlr')

PREFIX_GROUPS = [
    # Add prefixes here to make one plot per prefix. If multiple prefixes
    # match a filename, the longest prefix wins.
    # "PPO_LN_128",
    # "PPO_LN_RT_128",
    "PPO_LN",
    "PPO_LN_RT",
    "PPO-RTU_LN",
]

SMOOTHING_WINDOW = 100


def get_prefix_group(filename, prefixes):
    matches = [prefix for prefix in prefixes if filename.startswith(prefix)]
    if not matches:
        return None
    return max(matches, key=len)


def group_results_by_prefix(sub_results, prefixes):
    if not prefixes:
        return [(None, list(sub_results))]

    grouped_results = {prefix: [] for prefix in prefixes}
    for alg_result in sub_results:
        prefix = get_prefix_group(alg_result.filename, prefixes)
        if prefix is not None:
            grouped_results[prefix].append(alg_result)

    return [
        (prefix, alg_results)
        for prefix, alg_results in grouped_results.items()
        if alg_results
    ]


def safe_plot_name(value):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def smooth_curve(values, window):
    if window <= 1:
        return values

    values = np.asarray(values)
    window = min(window, values.shape[0])
    pad_left = (window - 1) // 2
    pad_right = window // 2
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection(Model=ExperimentModel, metrics=["ewm_reward"])
    results.paths = [path for path in results.paths if "hypers" not in path]
    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )
    for env, sub_results in results.groupby_directory(level=3):
        sub_results = list(sub_results)
        for prefix, grouped_sub_results in group_results_by_prefix(sub_results, PREFIX_GROUPS):
            fig, ax = plt.subplots(1, 1)
            if grouped_sub_results:
                ax.set_prop_cycle(color=select_colors(len(grouped_sub_results)))
            for alg_result in grouped_sub_results:
                alg = alg_result.filename
                print(alg)

                df = alg_result.load()
                print(df)
                if df is None:
                    continue

                report = Hypers.select_best_hypers(
                    df,
                    metric='mean_ewm_reward',
                    prefer=Hypers.Preference.high,
                    time_summary=TimeSummary.mean,
                    statistic=Statistic.mean,
                )

                exp = alg_result.exp

                xs, ys = extract_learning_curves(
                    df,
                    hyper_vals=report.best_configuration,
                    metric='ewm_reward',
                )

                xs = np.asarray(xs)
                ys = np.asarray(ys)
                assert np.all(np.isclose(xs[0], xs))

                res = curve_percentile_bootstrap_ci(
                    rng=np.random.default_rng(0),
                    y=ys,
                    statistic=Statistic.mean,
                    iterations=10000,
                )

                mean = smooth_curve(res.sample_stat, SMOOTHING_WINDOW)
                ci_low = smooth_curve(res.ci[0], SMOOTHING_WINDOW)
                ci_high = smooth_curve(res.ci[1], SMOOTHING_WINDOW)

                line = ax.plot(xs[0], mean, label=alg, linewidth=1.0)[0]
                ax.fill_between(
                    xs[0],
                    ci_low,
                    ci_high,
                    color=line.get_color(),
                    alpha=0.2,
                )

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

            path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
            plot_name = env if prefix is None else f"{env}-{safe_plot_name(prefix)}"
            save(
                save_path=f'{path}/plots',
                plot_name=plot_name,
                f=fig,
                height_ratio=2/3,
            )
