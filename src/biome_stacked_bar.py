import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from plotting_utils import (
    BIG_BIOME_COLORS,
    FONTSIZE,
    LABEL_MAP,
    TWO_BIOME_COLORS,
    WEATHER_BIOME_COLORS,
    PlottingArgumentParser,
    despine,
    filter_by_alg_aperture,
    get_biome_mapping,
    get_mapped_label,
    load_data,
    parse_plotting_args,
    save_plot,
)

# Mappings
SAMPLE_TYPE_MAP = {
    "999000:1000000:500": "Early",
    "4999000:5000000:500": "Middle",
    "9999000:10000000:500": "Late",
    "500000:1000000:500": "Early",
    "4500000:5000000:500": "Middle",
    "9500000:10000000:500": "Late",
}


def get_biome_plot_components(env, available_biomes, columns, window):
    biome_mapping = get_biome_mapping(env)
    available_biomes = sorted(available_biomes)

    if "TwoBiome" in env:
        biome_colors = TWO_BIOME_COLORS
        components = [
            (biome_mapping[b], [b]) for b in [0, -1, 1] if b in available_biomes
        ]
        sort_names = ["Morel", "Oyster"]
    elif "Weather" in env:
        biome_colors = WEATHER_BIOME_COLORS
        components = [
            (biome_mapping[b], [b]) for b in [0, -1, 1] if b in available_biomes
        ]
        sort_names = ["Hot", "Cold"]
    elif "ForagaxBig" in env:
        biome_colors = BIG_BIOME_COLORS
        primary_biomes = [0, 1, 2, 3]
        components = [
            (biome_mapping[b], [b]) for b in primary_biomes if b in available_biomes
        ]
        other_biomes = [b for b in available_biomes if b not in primary_biomes]
        if other_biomes:
            components.append(("Other", other_biomes))
        sort_names = [
            biome_mapping[b] for b in primary_biomes if b in available_biomes
        ]
    else:
        raise ValueError(f"Unknown biome mapping for environment: {env}")

    plot_components = []
    for idx, (name, biome_ids) in enumerate(components):
        metric_cols = [
            f"biome_{b}_occupancy_{window}"
            for b in biome_ids
            if f"biome_{b}_occupancy_{window}" in columns
        ]
        if metric_cols:
            plot_components.append(
                {
                    "name": name,
                    "metric": f"biome_component_{idx}",
                    "cols": metric_cols,
                    "sort_priority": name in sort_names,
                }
            )

    if not plot_components:
        raise ValueError(
            f"No biome occupancy columns found for environment {env} and window {window}"
        )

    return biome_colors, plot_components


def biome_component_mean_expr(component):
    cols = [pl.col(col) for col in component["cols"]]
    if len(cols) == 1:
        expr = cols[0].mean()
    else:
        expr = pl.sum_horizontal(cols).mean()
    return expr.fill_null(0.0).alias(component["metric"])


def occupancy_value(seed_data, seed, metric):
    value = seed_data[seed].get(metric) if metric is not None else None
    return 0.0 if value is None else value


def main():
    parser = PlottingArgumentParser(description="Plot biome occupancy as stacked bars.")
    parser.add_argument("--sample-types", nargs="*", help="Sample types to plot.")
    parser.add_argument(
        "--window", type=int, default=1000, help="Occupancy window size."
    )
    parser.add_argument(
        "--sort-seeds",
        action="store_true",
        help="Sort seeds by a metric before plotting.",
    )
    parser.add_argument(
        "--ylim", type=float, nargs=2, help="Set x-axis limits for the horizontal bars."
    )

    args = parse_plotting_args(parser)

    # Load and filter data
    df = load_data(args.experiment_path)
    df = filter_by_alg_aperture(df, args.filter_alg_apertures)

    if args.sample_types:
        df = df.filter(pl.col("sample_type").is_in(args.sample_types))
        sample_types_list = args.sample_types
    else:
        if "every" in df["sample_type"].unique().to_list():
            df = df.filter(pl.col("sample_type") == "every")
        df = df.with_columns(pl.lit("All").alias("sample_type"))
        sample_types_list = ["All"]

    if args.filter_seeds:
        df = df.filter(pl.col("seed").is_in(args.filter_seeds))

    env = df["env"][0]

    available_biomes = sorted(df["biome_id"].unique())
    biome_colors, plot_components = get_biome_plot_components(
        env, available_biomes, df.columns, args.window
    )
    biome_metrics = [component["metric"] for component in plot_components]
    biome_names = [component["name"] for component in plot_components]

    main_alg_apertures = sorted(df.select(["alg", "aperture"]).unique().rows())

    # Create figure
    nrows = len(main_alg_apertures)
    ncols = len(sample_types_list)
    fig, axs = plt.subplots(
        nrows, ncols, sharex=True, sharey=False, layout="constrained", squeeze=False
    )

    # Aggregate data
    agg_data = df.group_by(["alg", "aperture", "sample_type", "seed"]).agg(
        [biome_component_mean_expr(component) for component in plot_components]
    )

    # Plotting
    for i, (alg, aperture) in enumerate(main_alg_apertures):
        for j, sample_type in enumerate(sample_types_list):
            ax = axs[i, j]

            if aperture is None:
                plot_df = agg_data.filter(
                    (pl.col("alg") == alg)
                    & pl.col("aperture").is_null()
                    & (pl.col("sample_type") == sample_type)
                )
            else:
                plot_df = agg_data.filter(
                    (pl.col("alg") == alg)
                    & (pl.col("aperture") == aperture)
                    & (pl.col("sample_type") == sample_type)
                )

            if plot_df.is_empty():
                continue

            # Sort seeds if requested
            if args.sort_seeds:
                # Convert to dictionary format for sorting
                seed_data = {}
                for row in plot_df.iter_rows(named=True):
                    seed = row["seed"]
                    seed_data[seed] = {metric: row[metric] for metric in biome_metrics}

                # Sort seeds based on biome occupancy
                metric_by_name = {
                    component["name"]: component["metric"]
                    for component in plot_components
                }
                if "TwoBiome" in env:
                    morel_metric = metric_by_name.get("Morel")
                    oyster_metric = metric_by_name.get("Oyster")
                    sorted_seeds = sorted(
                        seed_data.keys(),
                        key=lambda s: (
                            -occupancy_value(seed_data, s, morel_metric),
                            occupancy_value(seed_data, s, oyster_metric),
                            str(s),
                        ),
                    )
                elif "Weather" in env:
                    hot_metric = metric_by_name.get("Hot")
                    cold_metric = metric_by_name.get("Cold")
                    sorted_seeds = sorted(
                        seed_data.keys(),
                        key=lambda s: (
                            -occupancy_value(seed_data, s, hot_metric),
                            occupancy_value(seed_data, s, cold_metric),
                            str(s),
                        ),
                    )
                elif "ForagaxBig" in env:
                    primary_metrics = [
                        component["metric"]
                        for component in plot_components
                        if component["sort_priority"]
                    ]
                    sorted_seeds = sorted(
                        seed_data.keys(),
                        key=lambda s: tuple(
                            -occupancy_value(seed_data, s, metric)
                            for metric in primary_metrics
                        )
                        + (str(s),),
                    )
                else:
                    sorted_seeds = plot_df["seed"].to_list()

                # Reorder plot_df based on sorted seeds
                seed_order = {seed: idx for idx, seed in enumerate(sorted_seeds)}
                plot_df = plot_df.with_columns(
                    pl.col("seed").replace(seed_order).alias("seed_order")
                )
                plot_df = plot_df.sort("seed_order")

            bottom = np.zeros(len(plot_df))

            # Create y-positions (0, 1, 2, ...) for each seed
            y_positions = np.arange(len(plot_df))

            for metric, name in zip(biome_metrics, biome_names, strict=True):
                values = np.nan_to_num(plot_df[metric].to_numpy(), nan=0.0)
                color = biome_colors[name]
                ax.barh(
                    y_positions,
                    values,
                    left=bottom,
                    label=name,
                    color=color,
                    height=1,
                    edgecolor=color,
                )
                bottom += values

            # Set y-axis limits and invert so first seed is on top
            ax.set_ylim(-0.5, len(plot_df) - 0.5)
            ax.invert_yaxis()

            despine(ax)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )
            ax.tick_params(
                axis="y", which="both", left=False, right=False, labelleft=False
            )
            ax.grid(False)

    # Formatting
    for i, (alg, aperture) in enumerate(main_alg_apertures):
        if aperture is not None:
            temp_label = f"{alg}:{aperture}"
        else:
            temp_label = alg
        label = get_mapped_label(temp_label, LABEL_MAP)

        if label:
            axs[i, 0].set_ylabel(label, rotation=0, ha="right", va="center")

    for j, sample_type in enumerate(sample_types_list):
        axs[-1, j].set_xlabel(SAMPLE_TYPE_MAP.get(sample_type, sample_type))

    if args.ylim:
        plt.setp(axs, xlim=args.ylim)
    else:
        plt.setp(axs, xlim=(0, 1))

    # Legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=biome_colors[n], label=n)
        for n in biome_names
    ]
    fig.legend(
        handles=legend_elements,
        loc="outside upper center",
        frameon=False,
        ncol=len(biome_names),
        fontsize=FONTSIZE,
    )

    plot_name = args.plot_name or f"{env}_biome_stacked_bar"
    save_plot(fig, args.experiment_path, plot_name, args.save_type)


if __name__ == "__main__":
    main()
