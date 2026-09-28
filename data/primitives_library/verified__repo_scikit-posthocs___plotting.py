from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from matplotlib import colors, pyplot
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar, ColorbarBase
from matplotlib.colors import ListedColormap
from pandas import DataFrame, Index, Series
from seaborn import heatmap


def sign_array(
    p_values: Union[List, np.ndarray, DataFrame], alpha: float = 0.05
) -> np.ndarray:
    sig_array = np.array(p_values, dtype=float)
    sig_array[sig_array == 0] = 1e-10
    sig_array[sig_array > alpha] = 0
    sig_array[(sig_array < alpha) & (sig_array > 0)] = 1
    np.fill_diagonal(sig_array, -1)
    return sig_array


def sign_table(
    p_values: Union[List, np.ndarray, DataFrame],
    lower: bool = True,
    upper: bool = True,
) -> Union[DataFrame, np.ndarray]:
    if not any((lower, upper)):
        raise ValueError("Either lower or upper triangle must be returned")

    if isinstance(p_values, DataFrame):
        table = p_values.copy()
    else:
        table = DataFrame(p_values, copy=True)

    nonsignificant = table > 0.05
    three_stars = (table < 0.001) & (table >= 0)
    two_stars = (table < 0.01) & (table >= 0.001)
    one_star = (table < 0.05) & (table >= 0.01)

    table = table.astype(str)
    table[nonsignificant] = "NS"
    table[three_stars] = "***"
    table[two_stars] = "**"
    table[one_star] = "*"

    for i in range(table.shape[0]):
        table.iat[i, i] = "-"

    if not lower:
        mask = ~np.tril(np.ones(table.shape, dtype=bool), -1)
        table = table.where(mask, other="")
    elif not upper:
        mask = ~np.triu(np.ones(table.shape, dtype=bool), 1)
        table = table.where(mask, other="")

    return table


def sign_plot(
    x: Union[List, np.ndarray, DataFrame],
    g: Union[List, np.ndarray, None] = None,
    flat: bool = False,
    labels: bool = True,
    cmap: Optional[List] = None,
    cbar_ax_bbox: Optional[Tuple[float, float, float, float]] = None,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Union[Axes, Tuple[Axes, Colorbar]]:
    for name in ("cbar", "vmin", "vmax", "center"):
        kwargs.pop(name, None)

    if isinstance(x, DataFrame):
        frame = x.copy()
    else:
        if g is None or len(g) == 0:
            g = np.arange(len(x))
        frame = DataFrame(x, index=Index(g), columns=Index(g), copy=True)

    dtype = frame.values.dtype
    if flat and not np.issubdtype(dtype, np.integer):
        raise ValueError("X should be a sign_array or DataFrame of integers")
    if not flat and not np.issubdtype(dtype, np.floating):
        raise ValueError("X should be an array or DataFrame of float p values")

    if not cmap:
        if flat:
            cmap = ["1", "#fbd7d4", "#1a9641"]
        else:
            cmap = ["1", "#fbd7d4", "#005a32", "#238b45", "#a1d99b"]

    if flat:
        for i in range(frame.shape[0]):
            frame.iat[i, i] = -1

        result = heatmap(
            frame,
            vmin=-1,
            vmax=1,
            cmap=ListedColormap(cmap),
            cbar=False,
            ax=ax,
            **kwargs,
        )
        if not labels:
            result.set_xlabel("")
            result.set_ylabel("")
        return result

    source = frame.values.copy()
    frame[(source < 0.001) & (source >= 0)] = 1
    frame[(source < 0.01) & (source >= 0.001)] = 2
    frame[(source < 0.05) & (source >= 0.01)] = 3
    frame[source >= 0.05] = 0

    for i in range(frame.shape[0]):
        frame.iat[i, i] = -1

    if len(cmap) != 5:
        raise ValueError("Cmap list must contain 5 items")

    result = heatmap(
        frame,
        vmin=-1,
        vmax=3,
        cmap=ListedColormap(cmap),
        cbar=False,
        ax=ax,
        **kwargs,
    )
    if not labels:
        result.set_xlabel("")
        result.set_ylabel("")

    cbar_ax = result.figure.add_axes(
        cbar_ax_bbox if cbar_ax_bbox is not None else (0.95, 0.35, 0.04, 0.3)
    )
    colorbar = ColorbarBase(
        cbar_ax,
        cmap=ListedColormap(cmap[2:] + [cmap[1]]),
        norm=colors.NoNorm(),
        boundaries=[0, 1, 2, 3, 4],
    )
    colorbar.set_ticks(
        list(np.linspace(0, 3, 4)),
        labels=["p < 0.001", "p < 0.01", "p < 0.05", "NS"],
    )
    colorbar.outline.set_linewidth(1)
    colorbar.outline.set_edgecolor("0.5")
    colorbar.ax.tick_params(size=0)

    return result, colorbar


def _bron_kerbosch(
    current_clique: Set,
    candidates: Set,
    excluded: Set,
    adj_matrix: DataFrame,
):
    if not candidates and not excluded:
        yield current_clique
        return

    while candidates:
        vertex = candidates.pop()
        neighbors = set(adj_matrix.columns[adj_matrix.loc[vertex].astype(bool)])
        yield from _bron_kerbosch(
            current_clique | {vertex},
            candidates & neighbors,
            excluded & neighbors,
            adj_matrix,
        )
        excluded.add(vertex)


def _find_maximal_cliques(adj_matrix: DataFrame) -> List[Set]:
    if not isinstance(adj_matrix, DataFrame):
        raise TypeError("adj_matrix must be a pandas DataFrame")

    if not adj_matrix.index.equals(adj_matrix.columns):
        raise ValueError("Adjacency matrix must be symmetric, indices and columns must match")

    if not adj_matrix.equals(adj_matrix.T):
        raise ValueError("Adjacency matrix must be symmetric")

    if not np.all(np.diag(adj_matrix.to_numpy()) == 1):
        raise ValueError("Adjacency matrix must have ones on the diagonal")

    return list(
        _bron_kerbosch(
            set(),
            set(adj_matrix.index),
            set(),
            adj_matrix,
        )
    )


def critical_difference_diagram(
    ranks: Series,
    sig_matrix: DataFrame,
    ax: Optional[Axes] = None,
    label_fmt_left: str = "{label} ({rank:.2g})",
    label_fmt_right: str = "({rank:.2g}) {label}",
    label_props: Optional[Dict] = None,
    marker_props: Optional[Dict] = None,
    elbow_props: Optional[Dict] = None,
    crossbar_props: Optional[Dict] = None,
    color_palette: Optional[Union[Dict, List]] = None,
    text_h_margin: float = 0.01,
) -> Dict[str, List]:
    if not isinstance(ranks, Series):
        ranks = Series(ranks)

    if not isinstance(sig_matrix, DataFrame):
        sig_matrix = DataFrame(sig_matrix, index=ranks.index, columns=ranks.index)

    if not ranks.index.is_unique:
        raise ValueError("Ranks index must be unique")

    if not sig_matrix.index.is_unique or not sig_matrix.columns.is_unique:
        raise ValueError("Significance matrix index and columns must be unique")

    if not ranks.index.equals(sig_matrix.index) or not ranks.index.equals(sig_matrix.columns):
        raise ValueError(
            "Ranks index and significance matrix index and columns must match"
        )

    if ax is None:
        ax = pyplot.gca()

    ranks = ranks.sort_values()
    sig_matrix = sig_matrix.loc[ranks.index, ranks.index]
    cliques = _find_maximal_cliques(sig_matrix)

    label_props = {} if label_props is None else dict(label_props)
    marker_props = {} if marker_props is None else dict(marker_props)
    elbow_props = {} if elbow_props is None else dict(elbow_props)
    crossbar_props = {} if crossbar_props is None else dict(crossbar_props)

    if color_palette is None:
        palette = {label: "k" for label in ranks.index}
    elif isinstance(color_palette, dict):
        palette = {label: color_palette.get(label, "k") for label in ranks.index}
    else:
        palette_values = list(color_palette)
        if not palette_values:
            palette = {label: "k" for label in ranks.index}
        else:
            palette = {
                label: palette_values[i % len(palette_values)]
                for i, label in enumerate(ranks.index)
            }

    n_methods = len(ranks)
    split = int(np.ceil(n_methods / 2.0))
    left_ranks = ranks.iloc[:split]
    right_ranks = ranks.iloc[split:]

    rank_min = float(ranks.min())
    rank_max = float(ranks.max())
    rank_span = rank_max - rank_min
    if rank_span == 0:
        rank_span = 1.0

    x_margin = rank_span * text_h_margin
    left_text_x = rank_min - x_margin
    right_text_x = rank_max + x_margin
    baseline = 0.5
    label_step = 0.12

    artists: Dict[str, List] = {
        "markers": [],
        "elbows": [],
        "labels": [],
        "crossbars": [],
    }

    marker_defaults = {"s": 36, "zorder": 3}
    marker_defaults.update(marker_props)

    elbow_defaults = {"linewidth": 1, "zorder": 2}
    elbow_defaults.update(elbow_props)

    crossbar_defaults = {"linewidth": 2, "color": "k", "zorder": 1}
    crossbar_defaults.update(crossbar_props)

    max_label_height = 0.0

    for position, (name, value) in enumerate(left_ranks.iloc[::-1].items()):
        y = baseline + (position + 1) * label_step
        max_label_height = max(max_label_height, y)
        color = palette[name]

        scatter_args = dict(marker_defaults)
        scatter_args.setdefault("color", color)
        artists["markers"].append(ax.scatter([value], [baseline], **scatter_args))

        elbow_args = dict(elbow_defaults)
        elbow_args.setdefault("color", color)
        artists["elbows"].extend(
            ax.plot(
                [value, value, left_text_x],
                [baseline, y, y],
                **elbow_args,
            )
        )

        text_args = dict(label_props)
        text_args.setdefault("ha", "right")
        text_args.setdefault("va", "center")
        text_args.setdefault("color", color)
        artists["labels"].append(
            ax.text(
                left_text_x,
                y,
                label_fmt_left.format(label=name, rank=value),
                **text_args,
            )
        )

    for position, (name, value) in enumerate(right_ranks.items()):
        y = baseline + (position + 1) * label_step
        max_label_height = max(max_label_height, y)
        color = palette[name]

        scatter_args = dict(marker_defaults)
        scatter_args.setdefault("color", color)
        artists["markers"].append(ax.scatter([value], [baseline], **scatter_args))

        elbow_args = dict(elbow_defaults)
        elbow_args.setdefault("color", color)
        artists["elbows"].extend(
            ax.plot(
                [value, value, right_text_x],
                [baseline, y, y],
                **elbow_args,
            )
        )

        text_args = dict(label_props)
        text_args.setdefault("ha", "left")
        text_args.setdefault("va", "center")
        text_args.setdefault("color", color)
        artists["labels"].append(
            ax.text(
                right_text_x,
                y,
                label_fmt_right.format(label=name, rank=value),
                **text_args,
            )
        )

    ordered_cliques = []
    for clique in cliques:
        if len(clique) > 1:
            values = ranks.loc[list(clique)]
            ordered_cliques.append((float(values.min()), float(values.max()), clique))
    ordered_cliques.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    occupied_levels: List[List[Tuple[float, float]]] = []
    for start, end, _ in ordered_cliques:
        level = 0
        while level < len(occupied_levels):
            overlaps = any(
                not (end < interval_start or start > interval_end)
                for interval_start, interval_end in occupied_levels[level]
            )
            if not overlaps:
                break
            level += 1

        if level == len(occupied_levels):
            occupied_levels.append([])
        occupied_levels[level].append((start, end))

        y = baseline - (level + 1) * label_step
        artists["crossbars"].extend(
            ax.plot(
                [start, end],
                [y, y],
                **crossbar_defaults,
            )
        )

    lower_limit = baseline - (len(occupied_levels) + 1) * label_step
    upper_limit = max(max_label_height + label_step, baseline + label_step)
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_xlim(left_text_x - rank_span * 0.1, right_text_x + rank_span * 0.1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)

    return artists