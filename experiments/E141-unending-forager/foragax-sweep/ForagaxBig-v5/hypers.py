import json
import sys
from pathlib import Path

import numpy as np
import rlevaluation.hypers as Hypers
from rlevaluation.config import data_definition
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import TimeSummary

ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from experiment.ExperimentModel import ExperimentModel  # noqa: E402
from experiment.hypers import update_best_config  # noqa: E402
from utils.results import ResultCollection  # noqa: E402


def main():
    results = ResultCollection(Model=ExperimentModel, metrics=["mean_ewm_reward"])
    results.paths = [path for path in results.paths if "hypers" not in path]
    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col="seed",
        time_col="frame",
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )
    env_reports = {}
    print(results.paths)
    for env, sub_results in results.groupby_directory(level=3):
        alg_reports = {}
        for alg_result in sub_results:
            alg = alg_result.filename

            df = alg_result.load()
            if df is None:
                continue
            df = df.sort(["seed", "id"])

            print(f"{env} {alg}")
            print(df)
            np.random.seed(0)
            report = Hypers.select_best_hypers(
                df,
                metric="mean_ewm_reward",
                prefer=Hypers.Preference.high,
                time_summary=TimeSummary.mean,
                statistic=Statistic.mean,
                threshold=0.05,
            )
            Hypers.pretty_print(report)
            best_configuration = {
                k: v
                for k, v in sorted(
                    report.best_configuration.items(),
                    key=lambda item: item[0],
                )
            }

            print(best_configuration)
            exp_path = Path(alg_result.exp_path)
            best_configuration_path = (
                exp_path.parent.parent / "hypers" / exp_path.parent.name / exp_path.name
            )
            best_configuration_path.parent.mkdir(parents=True, exist_ok=True)
            with open(
                best_configuration_path,
                "w",
            ) as f:
                json.dump(best_configuration, f, indent=4)

            alg_reports[alg] = {
                "result": alg_result,
                "report": report,
            }

            update_best_config(alg, report, exp_path)
        env_reports[env] = alg_reports


if __name__ == "__main__":
    main()
