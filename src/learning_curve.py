import logging
import math
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import polars as pl
import seaborn as sns
import tol_colors as tc

from annotate_plot import annotate_plot
from plotting_utils import (
    COLOR_MAP,
    LABEL_MAP,
    PlottingArgumentParser,
    despine,
    filter_by_alg_aperture,
    format_metric_name,
    get_mapped_label,
    get_ylabel_mapping,
    load_data,
    parse_plotting_args,
    save_plot,
)


def _max_bar_label_width(hue_order, args):
    max_label_len = max(
        (
            len(get_mapped_label(h, LABEL_MAP, disable_fov=args.disable_fov))
            for h in hue_order
        ),
        default=0,
    )
    return max_label_len * 0.15


def _default_bar_ratio(hue_order, args):
    return max(1.0, len(hue_order) / 3.0) if args.legend_on_bar else 1.0


def compute_figsize(num_metrics, hue_order, args):
    base_height = 6.0

    aspect_ratio = getattr(args, "aspect_ratio", None)
    if (
        aspect_ratio
        and not args.plot_avg
        and not args.plot_bar_only
        and not args.grid
        and not args.subplot_by_seed
    ):
        return (base_height * num_metrics * aspect_ratio, base_height * num_metrics), None

    base_width = 8.0 * getattr(args, "width_scale", 1.0)
    font_size = getattr(args, "font_size", None) or 24
    extra_width = (
        max(0.0, (font_size - 24) * 0.2)
        if args.horizontal_bars and args.legend_on_bar
        else 0.0
    )
    bar_ratio = None

    if args.plot_avg and not args.plot_bar_only:
        if args.horizontal_bars and args.legend_on_bar:
            bar_inner_width = 3.0
            total_bar_width = bar_inner_width + _max_bar_label_width(hue_order, args)
            bar_ratio = (total_bar_width / base_width) * 3.0
            fig_width = base_width + total_bar_width + extra_width
        else:
            bar_ratio = _default_bar_ratio(hue_order, args)
            fig_width = base_width + 2 * bar_ratio + extra_width
    elif args.plot_bar_only:
        if args.horizontal_bars and args.legend_on_bar:
            bar_inner_width = 3.0
            fig_width = (
                bar_inner_width
                + _max_bar_label_width(hue_order, args)
                + extra_width
                + 1.0
            )
        else:
            bar_ratio = _default_bar_ratio(hue_order, args)
            total_fig_width = base_width + 2 * bar_ratio
            fig_width = (
                total_fig_width * (bar_ratio / (3 + bar_ratio))
                + 2.0
                + extra_width
            )
    else:
        fig_width = base_width

    return (fig_width, base_height * num_metrics), bar_ratio


def main():
    parser = PlottingArgumentParser(description="Plot learning curves.")
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metric to plot on the y-axis (deprecated: use --metrics).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Metrics to plot on the y-axis. Multiple metrics will be plotted as subplots.",
    )
    parser.add_argument(
        "--width-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the base figure width.",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=float,
        default=None,
        help="Width:height ratio (e.g. 1.5 for 3:2). Overrides --width-scale for the plain curve plot.",
    )
    parser.add_argument(
        "--num-xticks",
        type=int,
        default=5,
        help="Approximate number of tick marks on the x-axis.",
    )
    parser.add_argument(
        "--num-yticks",
        type=int,
        default=10,
        help="Approximate number of tick marks on the y-axis.",
    )
    parser.add_argument(
        "--sample-type",
        type=str,
        default="every",
        help="Sample type to filter from the data.",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs="+",
        default=None,
        help="Y-axis limits for the plot. If one value is provided, it sets the upper limit.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Minimum frame to include in the plot",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Maximum frame to include in the plot",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Use legend instead of auto-labeling",
    )
    parser.add_argument(
        "--legend-on-bar",
        action="store_true",
        help="Place the legend as x-axis labels on the bar plot at 45 degree angle.",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default=None,
        help="Algorithm to normalize against.",
    )
    parser.add_argument(
        "--plot-all-seeds",
        action="store_true",
        help="Plot all individual seeds as separate lines instead of confidence intervals.",
    )
    parser.add_argument(
        "--subplot-by-seed",
        action="store_true",
        help="Create a grid of subplots, one per seed.",
    )
    parser.add_argument(
        "--vertical-lines",
        type=int,
        nargs="+",
        default=None,
        help="Frame numbers to draw vertical lines at.",
    )
    parser.add_argument(
        "--horizontal-lines",
        type=str,
        nargs="+",
        default=None,
        help="Horizontal lines to draw (e.g. upper bounds). "
             "Format: 'value:label' or just 'value'. "
             "Example: --horizontal-lines 0.95:Upperbound 0.5:Random",
    )
    parser.add_argument(
        "--plot-avg",
        action="store_true",
        help="Plot the Average Value alongside the learning curve.",
    )
    parser.add_argument(
        "--plot-bar-values",
        action="store_true",
        help="Plot exact values on the bar plot.",
    )
    parser.add_argument(
        "--horizontal-bars",
        action="store_true",
        help="Plot the Average Value bars horizontally. For use with --plot-avg.",
    )
    parser.add_argument(
        "--grid",
        type=str,
        nargs="+",
        default=None,
        help="Grid layout specification. Format: 'nrows,ncols cell1 cell2 ...'. "
             "Each cell can contain multiple algorithms separated by '+'. "
             "Example: '2,2 DQN EQRC DQN+EQRC PPO+A2C' creates a 2x2 grid where "
             "the first two cells show single algorithms, and the last two cells "
             "show comparisons of multiple algorithms.",
    )
    parser.add_argument(
        "--color-algs",
        type=str,
        nargs="+",
        default=None,
        help="List of algorithms to determine the color of each algorithm for consistent coloring across plots.",
    )

    parser.add_argument(
        "--disable-fov",
        action="store_true",
        help="Disable inserting the FOV (aperture) in the legend name.",
    )

    parser.add_argument(
        "--colors",
        type=str,
        nargs="+",
        default=None,
        help="Specific colors for each algorithm. Format: 'Alg:color' or 'Prefix:color:gradient'. "
             "If a prefix is used (e.g. DQN:darkgreen), all matching algorithms will receive a gradient based on that color. "
             "To use a Paul Tol color, prefix with 'tol:', e.g., 'tol:vibrant:blue' or just 'tol:blue' to pick from the active palette.",
    )

    parser.add_argument(
        "--curve-algs",
        type=str,
        nargs="+",
        default=None,
        help="List of algorithms to plot on the learning curve. If specified, only these algorithms will be plotted as lines, but the bar plot remains unaffected.",
    )

    parser.add_argument(
        "--plot-bar-only",
        action="store_true",
        help="Plot only the Average Value bar plot, without the learning curve.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Turn off all legends and annotations completely.",
    )

    args = parse_plotting_args(parser)

    # Parse horizontal lines specification
    horizontal_lines = []
    if args.horizontal_lines:
        for spec in args.horizontal_lines:
            if ":" in spec:
                value_str, label = spec.split(":", 1)
            else:
                value_str, label = spec, None
            horizontal_lines.append((float(value_str), label))

    # Handle backward compatibility: --metric takes precedence over --metrics default
    if args.metric:
        args.metrics = [args.metric]
    elif args.metrics is None:
        args.metrics = ["ewm_reward"]

    # Validate ylim
    if args.ylim and len(args.ylim) > 2:
        raise ValueError(
            f"--ylim expects 1 or 2 values (upper limit or [lower, upper]), got {len(args.ylim)}: {args.ylim}"
        )

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Load and filter data
    df = load_data(args.experiment_path)
    logger.info(f"After load_data: {df.select('alg').unique().to_pandas()}")
    logger.info(f"DataFrame columns: {df.columns}")

    df = filter_by_alg_aperture(df, args.filter_alg_apertures)
    logger.info(
        f"After filter_by_alg_aperture: {df.select('alg').unique().to_pandas()}"
    )

    logger.info(
        f"Sample types present: {df.select('sample_type').unique().to_pandas()}"
    )
    df = df.filter(pl.col("sample_type") == args.sample_type)
    logger.info(f"After sample_type filter: {df.select('alg').unique().to_pandas()}")

    if args.filter_algs:
        df = df.filter(pl.col("alg").is_in(args.filter_algs))

    hue_col = "alg"
    if args.filter_alg_apertures:
        df = df.with_columns(
            pl.when(pl.col("aperture").is_not_null())
            .then(pl.col("alg") + ":" + pl.col("aperture").cast(pl.Utf8))
            .otherwise(pl.col("alg"))
            .alias("alg_ap")
        )
        hue_col = "alg_ap"
        logger.info(
            f"After creating alg_ap: {df.select('alg_ap').unique().to_pandas()}"
        )

    if args.filter_seeds:
        df = df.filter(pl.col("seed").is_in(args.filter_seeds))

    df = df.filter(pl.col("frame") >= args.start_frame)
    if args.end_frame is not None:
        df = df.filter(pl.col("frame") <= args.end_frame)

    # # Patch temperature data for Weather environments (after filtering to reduce memory usage)
    env = df["env"][0]
    # if "Weather" in env:
    #     df = patch_temperature_data(df)
    #     logger.info("Patched temperature data into dataframe")

    logger.info(f"Final df shape: {df.shape}")
    logger.info(f"Final unique {hue_col}: {df.select(hue_col).unique().to_pandas()}")

    # Sort by order specified in arguments
    if args.filter_alg_apertures:
        alg_ap_order = {ap: i for i, ap in enumerate(args.filter_alg_apertures)}
        df = df.with_columns(pl.col("alg_ap").replace(alg_ap_order).alias("order_col"))
        df = df.sort("order_col")
    elif args.filter_algs:
        alg_order = {alg: i for i, alg in enumerate(args.filter_algs)}
        df = df.with_columns(pl.col("alg").replace(alg_order).alias("order_col"))
        df = df.sort("order_col")

    # Normalization
    if args.normalize:
        for metric in args.metrics:
            norm_df = df.filter(pl.col("alg") == args.normalize)
            norm_df = norm_df.select(
                [pl.col("frame"), pl.col("seed"), pl.col(metric).alias("norm_val")]
            )

            df = df.join(norm_df, on=["frame", "seed"], how="left")
            df = df.with_columns((pl.col(metric) / pl.col("norm_val")).alias(metric))
            df = df.drop("norm_val")

    # Plotting
    num_metrics = len(args.metrics)

    # Parse grid specification if provided
    grid_cells = None  # List of lists, each inner list contains algorithms for that cell
    grid_nrows = None
    grid_ncols = None
    present_hues = df.select(hue_col).unique().to_series().to_list()
    if args.color_algs:
        hue_order = [h for h in args.color_algs if h in present_hues]
        hue_order += [h for h in present_hues if h not in hue_order]
    elif args.filter_alg_apertures:
        hue_order = [h for h in args.filter_alg_apertures if h in present_hues]
        hue_order += [h for h in present_hues if h not in hue_order]
    elif args.filter_algs:
        hue_order = [h for h in args.filter_algs if h in present_hues]
        hue_order += [h for h in present_hues if h not in hue_order]
    else:
        hue_order = df.select(hue_col).unique(maintain_order=True).to_series().to_list()

    if args.grid:
        # Handle both formats:
        # 1. Separate args: --grid 2,2 DQN EQRC PPO A2C
        # 2. Single quoted string: --grid "2,2 DQN EQRC PPO A2C"
        grid_args = args.grid
        # If first element contains spaces, it was passed as a single quoted string
        if len(grid_args) == 1 and " " in grid_args[0]:
            grid_args = grid_args[0].split()

        # First element is 'nrows,ncols', rest are cell specifications
        grid_dims = grid_args[0].split(",")
        grid_nrows = int(grid_dims[0])
        grid_ncols = int(grid_dims[1])
        # Each cell spec can have multiple algorithms separated by '+'
        grid_cells = [cell.split("+") for cell in grid_args[1:]]

        if len(grid_cells) > grid_nrows * grid_ncols:
            raise ValueError(
                f"Too many cells ({len(grid_cells)}) for grid size {grid_nrows}x{grid_ncols}"
            )

    # Determine subplot layout
    if args.grid:
        assert grid_nrows is not None and grid_ncols is not None and grid_cells is not None
        fig, axes = plt.subplots(
            grid_nrows, grid_ncols, layout="constrained", figsize=(6 * grid_ncols, 4 * grid_nrows)
        )
        # Flatten axes array for easier indexing
        if grid_nrows == 1 and grid_ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
    elif args.subplot_by_seed:
        unique_seeds = df.select("seed").unique().sort("seed").to_series().to_list()
        num_seeds = len(unique_seeds)

        # Calculate grid dimensions
        ncols = math.ceil(math.sqrt(num_seeds))
        nrows = math.ceil(num_seeds / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, layout="constrained", figsize=(6 * ncols, 4 * nrows)
        )
        # Flatten axes array for easier indexing
        if num_seeds == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
    else:
        figsize, bar_ratio = compute_figsize(num_metrics, hue_order, args)

        if args.plot_avg and not args.plot_bar_only:
            assert bar_ratio is not None
            fig, axes = plt.subplots(
                num_metrics,
                2,
                layout="constrained",
                figsize=figsize,
                gridspec_kw={
                    'width_ratios': [3, bar_ratio],
                    'wspace': 0.15 if args.horizontal_bars else 0.05,
                },
            )
            if num_metrics == 1:
                axes = axes.reshape(1, 2)
        else:
            fig, axes = plt.subplots(
                num_metrics, 1, layout="constrained", figsize=figsize
            )
            if num_metrics == 1:
                axes = [axes]  # Make it a list for consistent handling


    # Create color palette matching the order in filter_alg_apertures
    vibrant_colors = list(tc.colorsets["vibrant"])
    muted_colors = list(tc.colorsets["muted"])

    combined_colors = vibrant_colors + [c for c in muted_colors if c not in vibrant_colors]

    # If we need more than 7 colors, use alternative palette from seaborn
    total_algs_to_color = len(args.color_algs) if args.color_algs else (len(args.filter_alg_apertures) if args.filter_alg_apertures else len(hue_order))
    if total_algs_to_color <= len(vibrant_colors):
        pass
    elif total_algs_to_color <= len(combined_colors):
        vibrant_colors = combined_colors
    else:
        vibrant_colors = sns.color_palette("husl", total_algs_to_color)

    if args.color_algs:
        # Map colors to algorithms in the order specified
        palette = {
            alg: vibrant_colors[i % len(vibrant_colors)]
            for i, alg in enumerate(args.color_algs)
        }
        # Ensure all items in hue_order have a color
        used_colors = len(args.color_algs)
        for h in hue_order:
            if h not in palette:
                palette[h] = vibrant_colors[used_colors % len(vibrant_colors)]
                used_colors += 1
    elif args.filter_alg_apertures:
        # Map colors to alg-aperture combinations in the order specified
        palette = {
            alg_ap: vibrant_colors[i % len(vibrant_colors)]
            for i, alg_ap in enumerate(args.filter_alg_apertures)
        }
    elif args.filter_algs:
        # Map colors: use COLOR_MAP if available, else fall back to cycling
        palette = {}
        fallback_idx = 0
        for alg in args.filter_algs:
            if alg in COLOR_MAP:
                palette[alg] = COLOR_MAP[alg]
            else:
                palette[alg] = vibrant_colors[fallback_idx % len(vibrant_colors)]
                fallback_idx += 1
    else:
        # Use default palette ordering
        palette = {}

    if args.colors:
        if palette is None:
            palette = {}
        for color_spec in args.colors:
            parts = color_spec.split(':')
            if len(parts) >= 2:
                is_gradient = False
                if parts[-1].lower() == 'gradient':
                    is_gradient = True
                    parts = parts[:-1]

                if len(parts) >= 4 and parts[-3].lower() == 'tol':
                    color_val = "tol:" + parts[-2] + ":" + parts[-1]
                    alg = ":".join(parts[:-3])
                elif len(parts) >= 3 and parts[-2].lower() == 'tol':
                    color_val = "tol:" + parts[-1]
                    alg = ":".join(parts[:-2])
                else:
                    color_val = parts[-1]
                    alg = ":".join(parts[:-1])

                # Extract proper base color
                if color_val.startswith("tol:"):
                    # user entered something like DQN:tol:vibrant:blue or DQN:tol:blue
                    tol_parts = color_val.split(":")
                    if len(tol_parts) == 3:
                        # e.g., tol:vibrant:blue
                        cset = tol_parts[1].lower()
                        cname = tol_parts[2].lower()
                        cset_obj = tc.colorsets.get(cset, tc.colorsets["vibrant"])
                        color = getattr(cset_obj, cname, vibrant_colors[0])
                    elif len(tol_parts) == 2:
                        # e.g., tol:blue (assume vibrant, then muted)
                        cname = tol_parts[1].lower()
                        if hasattr(tc.colorsets["vibrant"], cname):
                            color = getattr(tc.colorsets["vibrant"], cname)
                        else:
                            color = getattr(tc.colorsets["muted"], cname, vibrant_colors[0])
                    else:
                        color = color_val
                else:
                    color = color_val

                # Ensure color is string to appease type checks
                color_str = str(color)

                if is_gradient:
                    # also handle pt_dqn as a special case if looking for dqn
                    matches = [h for h in hue_order if alg in h or h.startswith(alg) or f"_{alg}" in h]
                    if matches:
                        if "-" in color_str:
                            c1, c2 = color_str.split("-", 1)
                            gradient = sns.blend_palette([c1, c2], n_colors=len(matches))
                        else:
                            # Avoid getting too close to pure white by generating a larger palette and taking the darker start
                            n_gradient = int(len(matches) * 1.5) + 2
                            gradient = sns.light_palette(color_str, n_colors=n_gradient, reverse=True)[:len(matches)]
                        for match, c in zip(matches, gradient):
                            palette[match] = c
                else:
                    # check if exact match exists
                    if alg in hue_order:
                        palette[alg] = color_str
                    elif any(h.startswith(alg) or f"_{alg}" in h for h in hue_order):
                        # check for exact match before prefix/gradient logic
                        exact_matches = [h for h in hue_order if h == alg or h == f"{alg}:" or h.startswith(f"{alg}:")]
                        if exact_matches:
                            for match in exact_matches:
                                palette[match] = color_str
                        else:
                            # try treating it as a prefix/gradient if they didn't specify gradient but it's a prefix
                            matches = [h for h in hue_order if h.startswith(alg) or f"_{alg}" in h]
                            if len(matches) > 1:
                                if "-" in color_str:
                                    c1, c2 = color_str.split("-", 1)
                                    gradient = sns.blend_palette([c1, c2], n_colors=len(matches))
                                else:
                                    n_gradient = int(len(matches) * 1.5) + 2
                                    gradient = sns.light_palette(color_str, n_colors=n_gradient, reverse=True)[:len(matches)]
                                for match, c in zip(matches, gradient):
                                    palette[match] = c
                            elif len(matches) == 1:
                                palette[matches[0]] = color_str.split("-")[0] if "-" in color_str else color_str
                    else:
                        palette[alg] = color_str.split("-")[0] if "-" in color_str else color_str

    if not palette:
        palette = None

    if args.grid:
        assert grid_cells is not None and grid_nrows is not None and grid_ncols is not None
        # Grid plot with specified algorithms per cell
        metric = args.metrics[0]  # Use the first metric for all grid cells
        if len(args.metrics) > 1:
            logger.warning(
                f"--grid mode only uses the first metric. Ignoring: {args.metrics[1:]}"
            )

        # Determine if we're using alg_ap based on cell specifications
        # If any cell contains ':', we're using alg_ap format (e.g., "DQN:5")
        use_alg_ap = any(":" in alg for cell in grid_cells for alg in cell)
        grid_hue_col = "alg_ap" if use_alg_ap else "alg"

        # Create alg_ap column if needed and it doesn't exist
        if use_alg_ap and "alg_ap" not in df.columns:
            df = df.with_columns(
                (pl.col("alg") + ":" + pl.col("aperture").cast(pl.Utf8)).alias("alg_ap")
            )

        for i, cell_algs in enumerate(grid_cells):
            ax = axes[i]

            # Build cell-specific palette for algorithms in this cell
            cell_palette = {
                alg: vibrant_colors[j % len(vibrant_colors)]
                for j, alg in enumerate(cell_algs)
            }

            # Filter data for algorithms in this cell
            cell_df_list = []
            missing_algs = []
            for alg in cell_algs:
                if use_alg_ap:
                    # Exact match for alg_ap (e.g., "DQN:5")
                    alg_df = df.filter(pl.col("alg_ap") == alg)
                else:
                    alg_df = df.filter(pl.col("alg") == alg)

                if alg_df.is_empty():
                    missing_algs.append(alg)
                else:
                    cell_df_list.append(alg_df)

            if not cell_df_list:
                logger.warning(f"No algorithms found for cell {i}, skipping")
                ax.text(0.5, 0.5, f"No data found", ha='center', va='center', transform=ax.transAxes)
                title = " + ".join(get_mapped_label(a, LABEL_MAP, disable_fov=args.disable_fov) for a in cell_algs)
                ax.set_title(title)
                despine(ax)
                continue

            if missing_algs:
                logger.warning(f"Algorithms not found in cell {i}: {missing_algs}")

            cell_df = pl.concat(cell_df_list)

            # Apply curve_algs filter if provided
            if args.curve_algs:
                cell_df = cell_df.filter(pl.col(grid_hue_col).is_in(args.curve_algs))
                cell_algs = [a for a in args.curve_algs if a in cell_algs]
                if cell_df.is_empty():
                    continue

                cell_order = {val: idx for idx, val in enumerate(cell_algs)}
                cell_df = cell_df.with_columns(pl.col(grid_hue_col).replace(cell_order).alias("curve_order_col")).sort("curve_order_col")

            # Configure lineplot based on whether to show all seeds or confidence intervals
            lineplot_kwargs = {
                "data": cell_df.to_pandas(),
                "x": "frame",
                "y": metric,
                "hue": grid_hue_col,
                "hue_order": [a for a in cell_algs if a not in missing_algs],
                "palette": cell_palette,
                "ax": ax,
                "legend": "full" if len(cell_algs) > 1 else False,
            }

            if args.plot_all_seeds:
                lineplot_kwargs["units"] = "seed"
                lineplot_kwargs["estimator"] = None
                lineplot_kwargs["alpha"] = 0.05
            else:
                lineplot_kwargs["errorbar"] = ("ci", 95)

            sns.lineplot(**lineplot_kwargs)

            # Formatting
            formatted_metric = format_metric_name(metric)
            ylabel_map = get_ylabel_mapping(env)
            ylabel = ylabel_map.get(formatted_metric, formatted_metric)
            if "ewm_reward" in metric or "rolling_reward" in metric:
                ylabel = "Average Reward"

            if args.normalize:
                ylabel = f"Normalized {ylabel}"

            # Set y-label only on leftmost column
            col_idx = i % grid_ncols
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")

            # Set x-label only on bottom row
            row_idx = i // grid_ncols
            if row_idx == grid_nrows - 1:
                ax.set_xlabel(r"Time steps $(\times 10^6)$")
            else:
                ax.set_xlabel("")

            # Title: show algorithm names (mapped)
            title = " + ".join(get_mapped_label(a, LABEL_MAP, disable_fov=args.disable_fov) for a in cell_algs if a not in missing_algs)
            ax.set_title(title)
            ax.set_xmargin(0)
            ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=args.num_xticks))
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{x / 1000000:g}")
            )
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=args.num_yticks))
            despine(ax)

            # Handle legend for multi-algorithm cells
            if len(cell_algs) > 1:
                handles, labels = ax.get_legend_handles_labels()
                mapped_labels = [get_mapped_label(label, LABEL_MAP, disable_fov=args.disable_fov) for label in labels]
                ax.legend(handles, mapped_labels, title=None, frameon=False, loc='best')

            if args.vertical_lines:
                for x in args.vertical_lines:
                    ax.axvline(x=x, color="grey", linestyle=":", alpha=0.5)

            for hline_val, hline_label in horizontal_lines:
                ax.axhline(y=hline_val, color="grey", linestyle="--", alpha=0.7,
                           label=hline_label)
                if hline_label:
                    ax.annotate(
                        hline_label,
                        xy=(1, hline_val),
                        xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=8, color="grey",
                    )

            if args.ylim:
                if len(args.ylim) == 1:
                    ax.set_ylim(top=args.ylim[0])
                else:
                    ax.set_ylim(args.ylim)

        # Share y-axis limits across all grid cells
        if not args.ylim:
            # Collect all y-limits from non-empty cells
            y_mins = []
            y_maxs = []
            for j in range(len(grid_cells)):
                ax = axes[j]
                if ax.has_data():
                    ylim = ax.get_ylim()
                    y_mins.append(ylim[0])
                    y_maxs.append(ylim[1])

            if y_mins and y_maxs:
                global_ylim = (min(y_mins), max(y_maxs))
                for j in range(len(grid_cells)):
                    axes[j].set_ylim(global_ylim)

        # Hide extra subplots if grid is not fully filled
        for j in range(len(grid_cells), grid_nrows * grid_ncols):
            axes[j].axis("off")
    elif args.subplot_by_seed:
        # Plot each seed in its own subplot
        for i, seed in enumerate(unique_seeds):
            ax = axes[i]
            seed_df = df.filter(pl.col("seed") == seed)

            curve_df = seed_df
            curve_hue_order = hue_order
            if args.curve_algs:
                curve_hue_order = [h for h in args.curve_algs if h in hue_order]
                curve_df = seed_df.filter(pl.col(hue_col).is_in(args.curve_algs))
                curve_order = {val: idx for idx, val in enumerate(curve_hue_order)}
                curve_df = curve_df.with_columns(pl.col(hue_col).replace(curve_order).alias("curve_order_col")).sort("curve_order_col")

            for metric in args.metrics:
                sns.lineplot(
                    data=curve_df.to_pandas(),
                    x="frame",
                    y=metric,
                    hue=hue_col,
                    hue_order=curve_hue_order,
                    palette=palette,
                    ax=ax,
                    legend="full" if i == 0 else False,
                )

            # Formatting
            if len(args.metrics) == 1:
                formatted_metric = format_metric_name(args.metrics[0])
                ylabel_map = get_ylabel_mapping(env)
                ylabel = ylabel_map.get(formatted_metric, formatted_metric)
                if "ewm_reward" in args.metrics[0] or "rolling_reward" in args.metrics[0]:
                    ylabel = "Average Reward"
                if args.normalize:
                    ylabel = f"Normalized {ylabel}"
            else:
                ylabel = "Value"

            ax.set_ylabel(ylabel)
            ax.set_xlabel(r"Time steps $(\times 10^6)$")
            ax.set_title(f"Seed {seed}")
            ax.set_xmargin(0)
            ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=args.num_xticks))
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{x / 1000000:g}")
            )
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=args.num_yticks))
            despine(ax)

            if args.vertical_lines:
                for x in args.vertical_lines:
                    ax.axvline(x=x, color="grey", linestyle=":", alpha=0.5)

            for hline_val, hline_label in horizontal_lines:
                ax.axhline(y=hline_val, color="grey", linestyle="--", alpha=0.7,
                           label=hline_label)
                if hline_label:
                    ax.annotate(
                        hline_label,
                        xy=(1, hline_val),
                        xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=8, color="grey",
                    )

            if args.ylim:
                if len(args.ylim) == 1:
                    ax.set_ylim(top=args.ylim[0])
                else:
                    ax.set_ylim(args.ylim)

        # Hide extra subplots if grid is not fully filled
        for j in range(num_seeds, len(axes)):
            axes[j].axis("off")

        # Handle legend
        if not args.legend:
            annotate_plot(axes[0], label_map=LABEL_MAP, disable_fov=args.disable_fov)
        else:
            handles, labels = axes[0].get_legend_handles_labels()
            mapped_labels = [get_mapped_label(label, LABEL_MAP, disable_fov=args.disable_fov) for label in labels]
            axes[0].legend(handles, mapped_labels, title=None, frameon=False)
    else:
        # Original plotting logic for metrics
        ax_auc = None
        for i, metric in enumerate(args.metrics):
            if args.plot_avg and not args.plot_bar_only:
                ax = axes[i][0]
                ax_auc = axes[i][1]
            elif args.plot_bar_only:
                ax = None
                ax_auc = axes[i] if num_metrics > 1 else axes[0]
            else:
                ax = axes[i] if num_metrics > 1 else axes[0]

            # Formatting label shared by curve and bar plots.
            formatted_metric = format_metric_name(metric)
            ylabel_map = get_ylabel_mapping(env)
            ylabel = ylabel_map.get(formatted_metric, formatted_metric)
            if "ewm_reward" in metric or "rolling_reward" in metric:
                ylabel = "Average Reward"

            if args.normalize:
                ylabel = f"Normalized {ylabel}"

            if not args.plot_bar_only:
                # Set up data for lineplot
                curve_df = df
                curve_hue_order = hue_order
                if args.curve_algs:
                    curve_hue_order = [h for h in args.curve_algs if h in hue_order]
                    curve_df = df.filter(pl.col(hue_col).is_in(args.curve_algs))
                    curve_order = {val: idx for idx, val in enumerate(curve_hue_order)}
                    curve_df = curve_df.with_columns(pl.col(hue_col).replace(curve_order).alias("curve_order_col")).sort("curve_order_col")

                # Configure lineplot based on whether to show all seeds or confidence intervals
                lineplot_kwargs = {
                    "data": curve_df.to_pandas(),
                    "x": "frame",
                    "y": metric,
                    "hue": hue_col,
                    "hue_order": curve_hue_order,
                    "palette": palette,
                    "ax": ax,
                    "legend": "full" if i == 0 else False,
                }

                if args.plot_all_seeds:
                    # Plot each seed as a separate line
                    lineplot_kwargs["units"] = "seed"
                    lineplot_kwargs["estimator"] = None
                    lineplot_kwargs["alpha"] = 0.05  # Make individual lines semi-transparent
                else:
                    # Plot mean with confidence intervals
                    lineplot_kwargs["errorbar"] = ("ci", 95)

                sns.lineplot(**lineplot_kwargs)

                ax.set_ylabel(ylabel)
                if i == num_metrics - 1:  # Only set x-label on the last subplot
                    ax.set_xlabel(r"Time steps $(\times 10^6)$")
                else:
                    ax.set_xlabel("")  # Remove x-label for non-last subplots
                ax.set_xmargin(0)
                ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=args.num_xticks))
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, _: f"{x / 1000000:g}")
                )
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=args.num_yticks))
                despine(ax)

                if args.vertical_lines:
                    for x in args.vertical_lines:
                        ax.axvline(x=x, color="grey", linestyle=":", alpha=0.5)

                for hline_val, hline_label in horizontal_lines:
                    ax.axhline(y=hline_val, color="grey", linestyle="--", alpha=0.7,
                               label=hline_label)
                    if hline_label:
                        ax.annotate(
                            hline_label,
                            xy=(1, hline_val),
                            xycoords=("axes fraction", "data"),
                            ha="right", va="bottom", fontsize=8, color="grey",
                        )

                if args.ylim:
                    if len(args.ylim) == 1:
                        ax.set_ylim(top=args.ylim[0])
                    else:
                        ax.set_ylim(args.ylim)

            if args.plot_avg or args.plot_bar_only:
                # Calculate the average over frames for each seed
                avg_df = df.group_by([hue_col, "seed"]).agg(pl.col(metric).mean().alias("avg"))

                # Check for frozen versions and map them to their base color
                is_frozen = [("frozen" in h.lower()) for h in hue_order]
                base_names = [re.sub(r"[-_ ]?frozen(?:[-_ ][a-zA-Z0-9]+)?", "", h, flags=re.IGNORECASE) for h in hue_order]
                base_names = [re.sub(r" \(Frozen\)", "", b, flags=re.IGNORECASE) for b in base_names]

                bar_palette = None
                if palette is not None:
                    bar_palette = {}
                    for idx_h, (h, base, frozen) in enumerate(zip(hue_order, base_names, is_frozen)):
                        if frozen and base in palette:
                            bar_palette[h] = palette[base]
                        else:
                            bar_palette[h] = palette.get(h, vibrant_colors[idx_h % len(vibrant_colors)])

                if args.horizontal_bars:
                    barplot = sns.barplot(
                        data=avg_df.to_pandas(),
                        y=hue_col,
                        x="avg",
                        hue=hue_col,
                        palette=bar_palette if bar_palette else palette,
                        order=hue_order,
                        ax=ax_auc,
                        capsize=.1,
                        errorbar=("ci", 95),
                        n_boot=1000,
                        legend=False,
                        dodge=False
                    )
                else:
                    barplot = sns.barplot(
                        data=avg_df.to_pandas(),
                        x=hue_col,
                        y="avg",
                        hue=hue_col,
                        palette=bar_palette if bar_palette else palette,
                        order=hue_order,
                        ax=ax_auc,
                        capsize=.1,
                        errorbar=("ci", 95),
                        n_boot=1000,
                        legend=False,
                        dodge=False
                    )

                # Extract valid patches (excluding zero-thickness artifacts if any)
                # We sort them by their primary coordinate to match hue_order
                valid_patches = [p for p in barplot.patches if getattr(p, 'get_height', lambda: 0)() != 0 or getattr(p, 'get_width', lambda: 0)() != 0]
                if args.horizontal_bars:
                    valid_patches.sort(key=lambda p: getattr(p, 'get_y', lambda: 0)())
                else:
                    valid_patches.sort(key=lambda p: getattr(p, 'get_x', lambda: 0)())

                # In case some patches were still filtered incorrectly, fallback to directly zipping if sizes match
                if len(valid_patches) != len(is_frozen):
                    valid_patches = barplot.patches[:len(is_frozen)]

                for patch, frozen in zip(valid_patches, is_frozen):
                    if frozen:
                        patch.set_hatch('//')
                        patch.set_edgecolor('white')

                if args.plot_bar_values:
                    # Calculate true means and bootstrapped CIs
                    import numpy as np

                    labels = []
                    lines_per_bar = 3 if len(ax_auc.lines) == len(valid_patches) * 3 else 1
                    for i, patch in enumerate(valid_patches):
                        if args.horizontal_bars:
                            val = patch.get_width()
                            err_data = ax_auc.lines[i * lines_per_bar].get_xdata()
                            err = (np.max(err_data) - np.min(err_data)) / 2
                        else:
                            val = patch.get_height()
                            err_data = ax_auc.lines[i * lines_per_bar].get_ydata()
                            err = (np.max(err_data) - np.min(err_data)) / 2

                        labels.append(f"{val:.3f} $\\pm$ {err:.3f}")

                    if len(ax_auc.containers) == 1:
                        ax_auc.bar_label(ax_auc.containers[0], labels=labels, padding=3)
                    else:
                        for i, container in enumerate(ax_auc.containers):
                            if len(container.patches) == 1 and i < len(labels):
                                ax_auc.bar_label(container, labels=[labels[i]], padding=3)
                            else:
                                ax_auc.bar_label(container, fmt='%.3f', padding=3)

                if args.horizontal_bars:
                    ax_auc.set_ylabel("")
                    ax_auc.set_xlabel(ylabel)
                    ax_auc.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
                    if args.legend_on_bar:
                        mapped_labels = [get_mapped_label(label, LABEL_MAP, disable_fov=args.disable_fov) for label in hue_order]
                        ax_auc.set_yticks(range(len(hue_order)))
                        ax_auc.set_yticklabels(mapped_labels)
                    else:
                        ax_auc.set_yticklabels([])
                        ax_auc.set_yticks([])
                else:
                    ax_auc.set_ylabel("")
                    ax_auc.set_xlabel("")
                    ax_auc.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
                    if args.legend_on_bar:
                        mapped_labels = [get_mapped_label(label, LABEL_MAP, disable_fov=args.disable_fov) for label in hue_order]
                        ax_auc.set_xticks(range(len(hue_order)))
                        ax_auc.set_xticklabels(mapped_labels, rotation=45, ha='right', va='center', rotation_mode='anchor')
                        ax_auc.tick_params(axis='x')
                    else:
                        ax_auc.set_xticklabels([])
                        ax_auc.set_xticks([])

                if args.ylim:
                    if args.horizontal_bars:
                        if len(args.ylim) == 1:
                            ax_auc.set_xlim(right=args.ylim[0])
                        else:
                            ax_auc.set_xlim(args.ylim)
                    else:
                        if len(args.ylim) == 1:
                            ax_auc.set_ylim(top=args.ylim[0])
                        else:
                            ax_auc.set_ylim(args.ylim)

                despine(ax_auc)

        # Handle legend
        if args.plot_avg and not args.plot_bar_only:
            ax_for_legend = axes[0][0]
        elif args.plot_bar_only:
            ax_for_legend = axes[0] if num_metrics > 1 else axes[0]
        else:
            ax_for_legend = axes[0] if num_metrics > 1 else axes[0]

        if args.no_legend:
            if ax_for_legend.get_legend():
                ax_for_legend.get_legend().remove()
            if (args.plot_avg or args.plot_bar_only) and ax_auc and ax_auc.get_legend():
                ax_auc.get_legend().remove()
        elif args.legend_on_bar and (args.plot_avg or args.plot_bar_only):
            leg = ax_for_legend.get_legend()
            if leg:
                leg.remove()
        elif not args.legend:
            if not args.plot_bar_only:
                annotate_plot(ax_for_legend, label_map=LABEL_MAP, disable_fov=args.disable_fov)
        else:
            handles, labels = ax_for_legend.get_legend_handles_labels()
            # If lineplot was not drawn, grab the handles from the barplot
            if not handles and args.plot_bar_only and ax_auc:
                handles, labels = ax_auc.get_legend_handles_labels()

            mapped_labels = [get_mapped_label(label, LABEL_MAP, disable_fov=args.disable_fov) for label in labels]
            if handles:
                legend_obj = ax_for_legend.legend(handles, mapped_labels, title=None, frameon=True, loc='best')
                legend_obj.get_frame().set_alpha(0.9)
                legend_obj.get_frame().set_facecolor('white')

    # Save plot
    if args.grid:
        # Flatten grid_cells to get all algorithm names for the filename
        all_algs = [alg for cell in grid_cells for alg in cell] if grid_cells else []
        algs_str = "_".join(all_algs) if all_algs else "grid"
        plot_name = args.plot_name or f"{env}_{algs_str}_grid"
    elif args.subplot_by_seed:
        plot_name = args.plot_name or f"{env}_by_seed"
    elif len(args.metrics) == 1:
        plot_name = args.plot_name or f"{env}_{args.metrics[0]}_curve"
    else:
        metrics_str = "_".join(args.metrics)
        plot_name = args.plot_name or f"{env}_{metrics_str}_curves"
    save_plot(fig, args.experiment_path, plot_name, args.save_type)


if __name__ == "__main__":
    main()
