import argparse
import concurrent.futures
import os
import sys
from itertools import product
from pathlib import Path

sys.path.append(os.getcwd() + "/src")

import polars as pl

from experiment.ExperimentModel import ExperimentModel
from utils.results import Result, ResultCollection


def _slurm_max_workers() -> int:
    cpt = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    nt = int(os.environ.get("SLURM_NTASKS", "0") or 0)
    if cpt and nt:
        return cpt * nt
    if nt:
        return nt
    on_node = int(os.environ.get("SLURM_CPUS_ON_NODE", "0") or 0)
    if on_node:
        return on_node
    return os.cpu_count() or 1


def process_alg_result(alg_result: Result, group, aperture):
    alg = alg_result.filename
    print(f"Loading: {group} {alg}")

    exp_path = Path(alg_result.exp_path)
    env = exp_path.parent.parent.name
    targets = [0, 1_000_000, 5_000_000, 10_000_000, 30_000_000]
    intervals = [1_000, 10_000, 100_000, 500_000, 1_000_000]
    n_samples = 500

    sample_types = []
    sample_types.append("every")
    for total, interval in product(targets, intervals):
        start, end = total - interval, total
        if start >= 0:
            sample_types.append((start, end, n_samples))
        start, end = total, total + interval
        sample_types.append((start, end, n_samples))
        start, end = total - interval, total + interval
        if start >= 0 and end <= max(targets):
            sample_types.append((start, end, n_samples))

    df = alg_result.load(sample_type=sample_types)  # type: ignore
    if df is None:
        return None

    df = df.with_columns(
        pl.lit(env).alias("env"),
        pl.lit(group).alias("group"),
        pl.lit(alg).alias("alg"),
        pl.lit(aperture).cast(pl.Int64).alias("aperture"),
    )
    print(f"Loaded: {group} {alg}")
    print(df)
    return df


def main(experiment_path: Path):
    output_path = (
        Path("results")
        / experiment_path.relative_to(Path("experiments"))
        / "data.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_path = experiment_path.resolve()

    results = ResultCollection(
        path=experiment_path,
        Model=ExperimentModel,
    )
    results.paths = [path for path in results.paths if "hypers" not in path]
    print("\n".join([str(path) for path in results.paths]))

    dfs = []
    for group, sub_results in results.groupby_directory(level=4):
        aperture = int(group) if group.isdigit() else None

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(_slurm_max_workers(), len(sub_results), 4)
        ) as executor:
            futures = [
                executor.submit(process_alg_result, alg_result, group, aperture)
                for alg_result in sub_results
            ]
            for future in concurrent.futures.as_completed(futures):
                df = future.result()
                if df is not None:
                    dfs.append(df)

    collected_dfs = []
    for df in dfs:
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        if not df.is_empty():
            collected_dfs.append(df)
    dfs = collected_dfs
    if not dfs:
        print("No data found to concatenate")
        return

    all_df = pl.concat(dfs, how="diagonal_relaxed")
    all_df = all_df.sort(["env", "group", "alg", "id", "frame"])
    # Defensive de-duplication: re-running training without clearing prior
    # results can leave two metadata variants of the same (run, frame) in the
    # collection (e.g. a config gained a hyperparameter between runs), which
    # would otherwise double every row and silently tighten CIs. Keep the last
    # (most recent) per natural key.
    key = ["env", "group", "alg", "aperture", "id", "seed", "frame", "sample_type"]
    key = [c for c in key if c in all_df.columns]
    before = all_df.height
    all_df = all_df.unique(subset=key, keep="last", maintain_order=True)
    if all_df.height != before:
        print(f"Dropped {before - all_df.height} duplicate rows (kept {all_df.height}).")
    all_df.write_parquet(output_path)
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process experiment data and save to parquet"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the experiment directory",
        default="experiments/E39-baselines/foragax/ForagaxTwoBiome-v3",
    )
    args = parser.parse_args()
    path = Path(args.path)
    main(path)
