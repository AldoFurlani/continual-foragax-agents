"""Normalized, parameter-count-weighted gradient-norm curves.

Reads the aggregated parquet written by ``process_data.py`` (columns
``grad_{l0,l1,l2}_<layer>`` and ``grad_nparams_<layer>``) and reproduces the
Figure-4c-style construction:

  1. per layer, per seed: divide the norm time series by its first-rollout
     value (first rollout -> 1.0 by construction);
  2. combine layers with a parameter-count-weighted average
     ``A(t) = sum_l n_l r_l(t) / sum_l n_l``;
  3. average across seeds (mean + bootstrap 95% CI).

Reading the parquet (not the raw .npz) means this works off the synced
results on a laptop, exactly like ``learning_curve.py``. One subplot per norm
type (l0, l1, l2); within each, a curve for the full network and one each for
the actor / critic sub-networks.

Usage:
    python src/grad_norm_curve.py <experiment_path> [--out <pdf>]
        [--norms l0 l1 l2] [--sample-type every]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

NORM_LABELS = {"l0": r"$\ell_0$", "l1": r"$\ell_1$", "l2": r"$\ell_2$"}
EPS = 1e-12


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


def _layers_in(df: pl.DataFrame) -> list:
    """Layers with gradient data in this (alg-filtered) frame. The parquet merges
    columns across algorithms, so a layer's columns can exist but be all-null for
    a given alg — keep only layers whose grad_nparams is actually populated."""
    layers = sorted(c[len("grad_l2_") :] for c in df.columns if c.startswith("grad_l2_"))
    return [
        layer
        for layer in layers
        if f"grad_nparams_{layer}" in df.columns
        and df[f"grad_nparams_{layer}"].drop_nulls().len() > 0
    ]


def _combined_matrix(df: pl.DataFrame, norm: str, layers: list):
    """(num_seeds, num_frames) matrix of the first-rollout-normalized,
    parameter-count-weighted combined curve, plus the frame axis."""
    den = 0.0
    terms = []
    for layer in layers:
        col = f"grad_{norm}_{layer}"
        n = float(df[f"grad_nparams_{layer}"][0])
        # per-seed baseline = value at the earliest frame for that seed
        base = df.group_by("id").agg(
            pl.col(col).sort_by("frame").first().alias("base")
        )
        wterm = f"wterm_{layer}"
        terms.append(
            df.join(base, on="id")
            .with_columns((n * pl.col(col) / (pl.col("base") + EPS)).alias(wterm))
            .select(["id", "frame", wterm])
        )
        den += n

    merged = terms[0]
    for t in terms[1:]:
        merged = merged.join(t, on=["id", "frame"])
    wcols = [f"wterm_{layer}" for layer in layers]
    merged = merged.with_columns((pl.sum_horizontal(wcols) / den).alias("A"))

    piv = merged.pivot(values="A", index="id", on="frame").sort("id")
    frame_cols = sorted((c for c in piv.columns if c != "id"), key=lambda c: int(c))
    mat = piv.select(frame_cols).to_numpy()
    frames = np.array([int(c) for c in frame_cols])
    return mat, frames


def _mean_ci(curves: np.ndarray, n_boot: int = 1000):
    mean = curves.mean(axis=0)
    n = curves.shape[0]
    if n < 2:
        return mean, None, None
    rng = np.random.default_rng(0)
    boot = curves[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return mean, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_path", type=str)
    parser.add_argument("--norms", nargs="+", default=["l0", "l1", "l2"])
    parser.add_argument("--sample-type", type=str, default="every")
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Plot the y-axis (grad norm) on a log scale.",
    )
    parser.add_argument(
        "--alg", type=str, default=None,
        help="Restrict to one algorithm; output then lands in plots/<alg>/.",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    pq = _parquet_path(Path(args.experiment_path))
    if not pq.exists():
        sys.exit(f"No parquet at {pq} — run process_data.py first.")
    df = pl.read_parquet(pq).filter(pl.col("sample_type") == args.sample_type)
    if args.alg:
        df = df.filter(pl.col("alg") == args.alg)
        if df.height == 0:
            print(f"No rows for alg {args.alg} — skipping.")
            return
    if df.height == 0:
        sys.exit(f"No rows with sample_type == {args.sample_type}")
    if not _layers_in(df):
        sys.exit("No grad_* columns in parquet — re-run training with grad-norm collection.")

    algs = sorted(df["alg"].unique().to_list())
    colors = {"full": "tab:orange", "actor": "tab:blue", "critic": "tab:red"}
    fig, axes = plt.subplots(
        len(args.norms), 1, figsize=(7, 3.2 * len(args.norms)), squeeze=False
    )

    for alg in algs:
        adf = df.filter(pl.col("alg") == alg)
        layers = _layers_in(adf)
        subsets = {
            "full": layers,
            "actor": [name for name in layers if name.startswith("actor")],
            "critic": [name for name in layers if name.startswith("critic")],
        }
        for ax, norm in zip(axes[:, 0], args.norms, strict=True):
            for name, subset in subsets.items():
                if not subset:
                    continue
                mat, frames = _combined_matrix(adf, norm, subset)
                x = frames / 1e6
                mean, lo, hi = _mean_ci(mat)
                style = "-" if name == "full" else "--"
                label = f"{alg}:{name}" if len(algs) > 1 else name
                ax.plot(x, mean, style, color=colors[name], label=label, lw=1.6)
                if lo is not None:
                    ax.fill_between(x, lo, hi, color=colors[name], alpha=0.15)
            ax.axhline(1.0, color="grey", lw=0.6, ls=":")
            if args.log_scale:
                ax.set_yscale("log")
            ax.set_ylabel(
                f"{NORM_LABELS.get(norm, norm)} grad norm\n(norm. to rollout 1)"
            )
    # Single legend on the first subplot (curves are identical across subplots).
    axes[0, 0].legend(fontsize=8, ncol=3)
    axes[-1, 0].set_xlabel(r"Time steps ($\times 10^6$)")
    fig.tight_layout()

    # Save alongside learning_curve.py's plots (under the experiments/ tree),
    # not next to the parquet (results/ tree), so all figures land together.
    exp_arg = Path(args.experiment_path)
    plots_dir = (exp_arg.parent if exp_arg.suffix == ".parquet" else exp_arg) / "plots"
    if args.alg:
        plots_dir = plots_dir / args.alg
    out = Path(args.out) if args.out else plots_dir / "grad_norm_curves.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
