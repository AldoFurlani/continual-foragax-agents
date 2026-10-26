"""Tables for E145: each agent across replay buffer sizes.

Usage: python summary.py <data.parquet>

Per agent and buffer size: the selected epsilon / alpha, mean reward over the
first and last 1M steps (+- 95% CI over seeds), last-1M biome rank, and how
many seeds collapsed -- last-1M reward < 0.01, or below half their best 1M
window. For DRQN, delta_q_first / delta_q_last averaged per 1M window.
"""

import re
import sys

import polars as pl

SIZES = ["1K", "10K", "100K", "1M"]
WINDOW = 1_000_000

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_fmt_str_lengths(40)


def main(path):
    schema = pl.scan_parquet(path).collect_schema()
    cols = ["alg", "seed", "frame", "rewards", "ewm_biome_rank_5", "epsilon", "optimizer.alpha"]
    dq_cols = [c for c in ("delta_q_first", "delta_q_last") if c in schema]
    df = pl.scan_parquet(path).select(cols + dq_cols).collect()

    parsed = df["alg"].map_elements(
        lambda a: (m := re.fullmatch(r"(.*)_B(\d+[KM])", a)) and [m[1], m[2]],
        return_dtype=pl.List(pl.String),
    )
    df = df.with_columns(
        parsed.list.get(0).alias("agent"),
        parsed.list.get(1).alias("buffer"),
        (pl.col("frame") // WINDOW).clip(0, 9).alias("window"),
    ).filter(pl.col("agent").is_not_null())
    order = pl.col("buffer").replace_strict(SIZES, list(range(len(SIZES))))

    per_window = df.group_by("agent", "buffer", "seed", "window").agg(
        pl.col("rewards").mean().alias("r"), pl.col("ewm_biome_rank_5").mean().alias("rank")
    )
    last = per_window.filter(pl.col("window") == 9)
    first = per_window.filter(pl.col("window") == 0)
    peak = per_window.group_by("agent", "buffer", "seed").agg(pl.col("r").max().alias("peak"))
    seeds = last.join(peak, on=["agent", "buffer", "seed"]).join(
        first.select("agent", "buffer", "seed", pl.col("r").alias("r_first")),
        on=["agent", "buffer", "seed"],
    )

    def ci(c):
        return 1.96 * pl.col(c).std() / pl.len().sqrt()

    table = (
        seeds.group_by("agent", "buffer")
        .agg(
            pl.len().alias("seeds"),
            pl.col("r_first").mean().round(3).alias("r_first1M"),
            pl.col("r").mean().round(3).alias("r_last1M"),
            ci("r").round(3).alias("r_ci"),
            pl.col("rank").mean().round(2).alias("rank_last1M"),
            ci("rank").round(2).alias("rank_ci"),
            (pl.col("r") < 0.01).sum().alias("dead"),
            (pl.col("r") < 0.5 * pl.col("peak")).sum().alias("below_half_peak"),
        )
        .join(
            df.group_by("agent", "buffer").agg(
                pl.col("epsilon").first(), pl.col("optimizer.alpha").first().alias("alpha")
            ),
            on=["agent", "buffer"],
        )
        .sort("agent", order)
    )
    print("Reward / rank by buffer size (lower rank is better)")
    print(table)

    if dq_cols:
        dq = (
            df.filter(pl.col("agent").str.starts_with("DRQN"))
            .group_by("agent", "buffer", "window")
            .agg(*[pl.col(c).cast(pl.Float32).mean().round(4) for c in dq_cols])
        )
        if len(dq):
            for c in dq_cols:
                print(f"\nDRQN {c}, mean per 1M window")
                print(dq.pivot(on="window", index=["agent", "buffer"], values=c, sort_columns=True).sort("agent", order))


if __name__ == "__main__":
    main(sys.argv[1])
