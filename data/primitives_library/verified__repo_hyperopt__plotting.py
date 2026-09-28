"""
Tools for plotting results stored in Hyperopt trial collections.
"""

import pickle

import numpy as np

from . import base
from .base import miscs_to_idxs_vals

__authors__ = "James Bergstra"
__license__ = "3-clause BSD License"
__contact__ = "github.com/hyperopt/hyperopt"

default_status_colors = {
    base.STATUS_NEW: "k",
    base.STATUS_RUNNING: "g",
    base.STATUS_OK: "b",
    base.STATUS_FAIL: "r",
}


def main_plot_history(trials, do_show=True, status_colors=None, title="Loss History"):
    import matplotlib.pyplot as plt

    colors_by_status = default_status_colors if status_colors is None else status_colors
    points = [
        (loss, colors_by_status[status])
        for loss, status in zip(trials.losses(), trials.statuses())
        if loss is not None
    ]
    ys, colors = zip(*points)

    plt.scatter(range(len(ys)), ys, c=colors)
    plt.xlabel("time")
    plt.ylabel("loss")

    best_error = trials.average_best_error()
    print("avg best error:", best_error)
    plt.axhline(best_error, c="g")
    plt.title(title)

    if do_show:
        plt.show()


def main_plot_histogram(trials, do_show=True, title="Loss Histogram"):
    import matplotlib.pyplot as plt

    records = [
        (spec, loss, status, default_status_colors[status])
        for spec, loss, status in zip(trials.specs, trials.losses(), trials.statuses())
        if loss is not None
    ]
    xs, ys, statuses, colors = zip(*records)

    print("Showing Histogram of %i jobs" % len(ys))
    plt.hist(ys)
    plt.xlabel("loss")
    plt.ylabel("frequency")
    plt.title(title)

    if do_show:
        plt.show()


def main_plot_vars(
    trials,
    do_show=True,
    fontsize=10,
    colorize_best=None,
    columns=5,
    arrange_by_loss=False,
):
    import matplotlib.pyplot as plt

    idxs, vals = miscs_to_idxs_vals(trials.miscs)
    losses = trials.losses()
    finite_losses = [loss for loss in losses if loss not in (None, float("inf"))]
    sorted_loss_indices = np.argsort(finite_losses)

    if colorize_best is None:
        colorize_threshold = finite_losses[sorted_loss_indices[0]] - 1
    else:
        colorize_threshold = finite_losses[sorted_loss_indices[colorize_best + 1]]

    loss_min = min(finite_losses)
    loss_max = max(finite_losses)
    print("finite loss range", loss_min, loss_max, colorize_threshold)

    losses_for_tid = dict(zip(trials.tids, losses))

    def rainbow_color(loss):
        if loss is None:
            return (1, 1, 1)

        position = 4 * (loss - loss_min) / (loss_max - loss_min + 0.0001)
        if position < 1:
            return (position, 0, 0)
        if position < 2:
            return (2 - position, position - 1, 0)
        if position < 3:
            return (0, 3 - position, position - 2)
        return (0, 0, 4 - position)

    def grayscale_color(loss):
        if loss in (None, float("inf")):
            return (1, 1, 1)

        position = (loss - loss_min) / (loss_max - loss_min + 0.0001)
        if loss < colorize_threshold:
            return (0.0, 1.0 - position, 0.0)
        return (position, position, position)

    labels = list(idxs.keys())
    titles = labels
    ordering = np.argsort(titles)

    column_count = min(columns, len(labels))
    row_count = int(np.ceil(len(labels) / float(column_count)))

    for plot_number, variable_number in enumerate(ordering):
        label = labels[variable_number]
        plt.subplot(row_count, column_count, plot_number + 1)

        tick_positions, tick_labels = plt.xticks()
        plt.xticks(tick_positions, [""] * len(tick_positions))

        if arrange_by_loss:
            x_values = [losses_for_tid[trial_id] for trial_id in idxs[label]]
        else:
            x_values = idxs[label]

        if "log" in label:
            y_values = np.log(vals[label])
        else:
            y_values = vals[label]

        plt.title(titles[variable_number], fontsize=fontsize)
        point_colors = [
            grayscale_color(losses_for_tid[trial_id]) for trial_id in idxs[label]
        ]

        if len(y_values):
            plt.scatter(x_values, y_values, c=point_colors)

        if "log" in label:
            y_ticks, y_tick_labels = plt.yticks()
            plt.yticks(y_ticks, ["%.2e" % np.exp(value) for value in y_ticks])

    if do_show:
        plt.show()


def main_plot_1D_attachment(
    trials,
    attachment_name,
    do_show=True,
    colorize_by_loss=True,
    max_darkness=0.5,
    num_trails=None,
    preprocessing_fn=lambda x: x,
    line_width=0.1,
):
    """
    Plot one-dimensional attachment data associated with trials.

    A legend is included only when fewer than ten trial lines are plotted.
    """
    import matplotlib.pyplot as plt

    plt.title(attachment_name)

    defined_losses = [loss for loss in trials.losses() if loss is not None]
    lowest_loss = min(defined_losses)
    highest_loss = max(defined_losses)

    if num_trails is None:
        selected_trials = trials
    else:
        ordered_trials = sorted(
            filter(lambda trial: "loss" in trial["result"], trials),
            key=lambda trial: trial["result"]["loss"],
        )
        selected_trials = [
            ordered_trials[index]
            for index in np.linspace(
                0,
                len(ordered_trials),
                num_trails,
                endpoint=False,
                dtype=int,
            )
        ]

    for trial in selected_trials:
        attachments = trials.trial_attachments(trial)
        if attachment_name not in attachments:
            continue

        data = np.squeeze(np.asanyarray(pickle.loads(attachments[attachment_name])))
        if len(data.shape) != 1:
            continue

        data = preprocessing_fn(data)

        if colorize_by_loss:
            color = (
                0.0,
                0.0,
                0.0,
                max_darkness
                * (trial["result"]["loss"] - lowest_loss)
                / (highest_loss - lowest_loss),
            )
        else:
            color = None

        plt.plot(
            data,
            color=color,
            linewidth=line_width,
            label="loss: {:.5}".format(trial["result"]["loss"]),
        )

    if do_show:
        if len(selected_trials) < 10:
            plt.legend()
        plt.show()