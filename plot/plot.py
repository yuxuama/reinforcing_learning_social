"""plot/plot.py
Implement all the plot functions not evolving datasets"""

from plot.plot_init import *
from scipy.optimize import curve_fit

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
    fig, ax = plt.subplots(layout[0], layout[1], figsize=(10, 6.5), constrained_layout=True)
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
                xlabel="Expected probability",
                box_aspect=1
            )
    fig.suptitle("Average histogram of 'trust' per phenotype")
    return ax

def plot_xhi_by_phenotype(xhi_means, **plot_kwargs):
    """Plot all mean xhis for each phenotype and the corresponding eta fit"""
    possible_phenotype = list(xhi_means.keys())
    possible_phenotype.append(possible_phenotype.pop(0))
    fig_layout = [(1, 2), (1, 3), (2, 3), (2, 3), (2, 3)]
    layout = fig_layout[len(possible_phenotype) - 2]
    fig, ax = plt.subplots(layout[0], layout[1], figsize=(10, 6.5), constrained_layout=True)
    model = lambda j, eta: (np.exp(eta * j) - 1) / (np.exp(eta) - 1)
    for i in range(layout[0]):
        for j in range(layout[1]):
            if layout[0] == 1:
                selector = j
            else:
                selector = (i, j)
            index = i * layout[1] + j
            ph = possible_phenotype[index]
            xhi = xhi_means[ph]
            t_norm = np.linspace(0, 1, xhi.size)
            popt, _ = curve_fit(model, t_norm, xhi)

            ax[selector].plot(t_norm, xhi, "+", color="tab:blue", label="Mean", **plot_kwargs)
            ax[selector].plot(t_norm, model(t_norm, popt[0]), color="tab:orange", label="Fit ($\eta$ = {})".format(round(popt[0], 2)))
            ax[selector].set_title(ph)
            ax[selector].legend()
            ax[selector].set_box_aspect(1)
    fig.suptitle("Average xhi by phenotype")
    return ax

def plot_cooperation_per_phenotype(response, phenotype, expected=False):
    """Plot the probability of response per phenotype for each type of game played against
    the phenotype `phenotype`"""

    games = ["PD", "SH", "SD", "HG"]
    possible_phenotype = sorted(list(response.keys()))

    if expected:
        data_selector = []
        for i in range(4):
            data_selector.append("pe" + games[i])
    else:
        data_selector = []
        for i in range(4):
            data_selector.append("p" + games[i])
    
    fig_layout = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    layout = fig_layout[len(possible_phenotype) - 1]
    fig, ax = plt.subplots(layout[0], layout[1], figsize=(11, 2.5), sharey=True)
    for i in range(layout[0]):
        for j in range(layout[1]):
            if layout[0] == 1:
                selector = j
            else:
                selector = (i, j)
            index = i * layout[1] + j
            if index == 0:
                label = "probability"
                if expected:
                    label = "expected " + label 
                ax[selector].set_ylabel(label)
            key = possible_phenotype[index]
            data = np.zeros(4)
            for k in range(4):
                data[k] = response[key][phenotype][data_selector[k]]
            
            ax[selector].bar(games, data, align='center', color="tab:blue")
            reverse = 1 - data
            ax[selector].bar(games, reverse, align='center', bottom=data, color="tab:red")
            ax[selector].set_title(key)
            ax[selector].set_ylim([0, 1])
            ax[selector].set_box_aspect(1)
    
    if expected:
        title = "Expected probability versus {}".format(phenotype)
    else:
        title = "Probability versus {}".format(phenotype)
    
    fig.suptitle(title)
    return ax

