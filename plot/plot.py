"""plot/plot.py
Implement all the plot functions not evolving datasets"""

from plot.plot_init import *

def plot_histogram(data, bins, ax=None, **plot_kwargs):
    """Plot the histogram with `data` corresponding to the content of the `bins` on the axis `ax`.
    If `ax` is None then create a subplot (1, 1, 1)"""
    if ax is None:
        ax = plt.subplot(1, 1, 1)
    placeholder = bins[:(len(bins) - 1)]
    ax.hist(placeholder, bins=bins, weights=data, **plot_kwargs)
    return ax

def plot_histogram_per_phenotype(mean_histogram, bins, **hist_kwargs):
    """Plot the mean histogram for each phenotype"""
    possible_field = []
    for key in mean_histogram.keys():
        if key != "Global":
            possible_field.append(key)
    possible_field.sort()
    possible_field.append("Global")

    fig_layout = [(1, 2), (1, 3), (2, 3), (2, 3), (2, 3)]
    layout = fig_layout[len(possible_field) - 2]
    fig, ax = plt.subplots(layout[0], layout[1], figsize=(8, 6))
    for i in range(layout[0]):
        for j in range(layout[1]):
            if layout[0] == 1:
                selector = j
            else:
                selector = (i, j)
            index = i * layout[1] + j
            data = mean_histogram[possible_field[index]]
            plot_histogram(data, bins, ax=ax[selector], **hist_kwargs)
            ax[selector].set(
                title=possible_field[index],
                ylabel="Occurence",
                xlabel="Expected probability"
            )
    fig.suptitle("Average histogram of 'trust' per phenotype")
            