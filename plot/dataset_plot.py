"""plot/dataset_plot.py
Implement all the plot that require datasets"""

from plot.plot_init import *
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

def plot_hist_by_phenotype(dataset, quantity, **plot_kwargs):
    """Generate appropriate layour for phenotype ploting"""
    dta = dataset.group_by("Phenotype").aggregate(quantity)
    global_dta = dataset.aggregate(quantity)
    possible_phenotype = sorted(list(dta.keys()))
    fig_layout = [(1, 2), (1, 3), (2, 3), (2, 3), (2, 3)]
    layout = fig_layout[dta.size - 1]
    fig, ax = plt.subplots(layout[0], layout[1], figsize=(8, 6))
    for i in range(layout[0]):
        for j in range(layout[1]):
            if layout[0] == 1:
                selector = j
            else:
                selector = (i, j)
            index = i * layout[1] + j
            if index == dta.size:
                data = global_dta.get_item(quantity).get_all_item()
                title = "Global"
            else:
                data = dta.get_item(possible_phenotype[index]).get_item(quantity).get_all_item()
                title = possible_phenotype[index]
            data = list(data.values())
            n, _, _ = ax[selector].hist(data, **plot_kwargs)
            maxi = np.max(n)
            median = np.median(data)
            ax[selector].vlines(median, 0, maxi*1.1, colors="k", linestyles="dashed", label="Median: {}".format(round(median, 2)))
            ax[selector].set_title(title)
            ax[selector].set_ylim([0, maxi*1.1])
            ax[selector].legend()
    fig.suptitle("Histogram of {0} @ inter {1}".format(quantity, dataset.niter))
    return ax

def plot_diadic_pattern(dataset, **plot_kwargs):
    """Plot bar graph with diadic pattern repartition per phenotype"""
    possible_fields = list(dataset.get_item(".").keys())
    dta = dataset.aggregate(possible_fields)
    fig_layout = [(1, 2), (1, 3), (2, 3), (2, 3), (2, 3)]
    layout = fig_layout[len(possible_fields) - 2]
    fig, ax = plt.subplots(layout[0], layout[1], figsize=(10, 6))
    for i in range(layout[0]):
        for j in range(layout[1]):
            if layout[0] == 1:
                selector = j
            else:
                selector = (i, j)
            index = i * layout[1] + j
            if index == dta.size:
                title = "Global"
            else:
                title = possible_fields[index]
            data = dta.get_item(possible_fields[index]).get_all_item()
            ax[selector].bar(list(data.keys()), list(data.values()), **plot_kwargs)
            ax[selector].set_title(title)
            ax[selector].set_ylabel("Occurences")
    fig.suptitle("Diadic pattern @ iter {}".format(dataset.niter))
    return ax

def plot_bar_diadic_pattern(dataset, **plot_kwargs):
    possible_phenotype = list(dataset.get_item(".").keys())
    dta = dataset.aggregate(possible_phenotype)
    bottom = np.zeros(4)
    ax = plt.subplot(1, 1, 1)
    for i in range(len(possible_phenotype)-1):
        ph = possible_phenotype[i]
        data = dta.get_item(ph).get_all_item()
        ax.bar(list(data.keys()), np.array(list(data.values()))/2, bottom=bottom, label=ph, **plot_kwargs)
        bottom += np.array(list(data.values()))/2
    ax.set_title("Diadic pattern @ inter {}".format(dataset.niter))
    ax.set_ylabel("Occurences")
    plt.legend()

def plot_bar_diadic_pattern_from_hist(freq_dict_hist, diadic_list, **plot_kwargs):
    bottom = np.zeros(4)
    ax = plt.subplot(1, 1, 1)
    for ph in freq_dict_hist.keys():
        if ph != "Number":
            data = freq_dict_hist[ph] / 2
            ax.bar(diadic_list, data, bottom=bottom, label=ph, **plot_kwargs)
        bottom += data
    ax.set_title("Diadic pattern from hist")
    ax.set_ylabel("Occurences")
    plt.legend()

def plot_phenotype_combination_per_link(link_type, combination_dt, th=0):
    """Plot for the linl with `link_type` (among `'.'`, `'->'`, `'<-'`, `'<->'`) the possible combination of phenotype calculated in
    `combination_dt` with number of occurences above `th`"""
    dtga = combination_dt.aggregate("Number")
    data = dtga.get_item(link_type).get_item("Number").get_all_item()
    data_order = sorted(data.items(), key=lambda x: x[1])
    plot_combination = []
    plot_data = []
    for i in range(len(data_order)):
        if data_order[i][1] > th:
            plot_combination.append(data_order[i][0])
            plot_data.append(data_order[i][1])
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(range(len(plot_data)), plot_data)
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(plot_combination)
    ax.set_title("Phenotype combinations distribution for {0} with threshold {1}".format(link_type, th))
    ax.set_ylabel("Occurence")
    return ax

def plot_triadic_pattern(triadic_dataset, selector="Number", triangle_only=False, **plot_kwargs):
    data = triadic_dataset.aggregate(selector).get_item(selector).get_all_item()
    triangle_names = list(data.keys())

    def get_image(name):
        path = r"C:\Users\Matthieu\Documents\_Travail\Stages\Stage M1\Workspace\image\triadic_{}.png".format(name)
        im = plt.imread(path)
        return im

    def offset_image(coord, name, ax):
        img = get_image(name)
        im = OffsetImage(img, zoom=0.2)
        im.image.axes = ax
        ab = AnnotationBbox(im, (coord, 0),  xybox=(0., -16.), frameon=False,
                            xycoords='data',  boxcoords="offset points", pad=0)
        ax.add_artist(ab)

    _, ax = plt.subplots(1, 1, figsize=(8, 5))

    data = np.array(list(data.values()))
    if triangle_only:
        data = np.array(list(data.values()))[3::]

    ax.bar(range(0, len(data)*2, 2), data, width=1, align="center", **plot_kwargs)
    ax.tick_params(axis='x', which='both', labelbottom=False, top=False, bottom=False)
    ax.set_ylabel("Occurences")
    if selector == "Number":
        title = "Global"
    else:
        title = selector
    ax.set_title("Triadic pattern frequency for {0} @ inter {1}".format(title, triadic_dataset.niter))

    for i in range(0, len(data)*2, 2):
        if triangle_only:
            offset_image(i, triangle_names[i//2+3], ax)
        else:
            offset_image(i, triangle_names[i//2], ax)
    
    return ax

def plot_triadic_pattern_phenotype(triadic_dataset, parameters, triangle_only=False, **plot_kwargs):
    possible_fields = list(parameters["Strategy distributions"].keys())
    possible_fields.append("Number")
    data = triadic_dataset.aggregate(possible_fields)
    triangle_names = list(data.get_item("Number").get_all_item().keys())

    def get_image(name):
        path = r"C:\Users\Matthieu\Documents\_Travail\Stages\Stage M1\Workspace\image\triadic_{}.png".format(name)
        im = plt.imread(path)
        return im

    def offset_image(coord, name, ax):
        img = get_image(name)
        im = OffsetImage(img, zoom=0.2)
        im.image.axes = ax
        ab = AnnotationBbox(im, (coord, 0),  xybox=(0., -16.), frameon=False,
                            xycoords='data',  boxcoords="offset points", pad=0)
        ax.add_artist(ab)

    _, ax = plt.subplots(1, 1, figsize=(8, 5))

    bottom = np.zeros(16)
    if triangle_only:
        bottom = bottom[3::]
    for i in range(len(possible_fields) - 1):
        ph_data = data.get_item(possible_fields[i]).get_all_item()
        values = np.array(list(ph_data.values())) / 3
        if triangle_only:
            values = values[3::]
        ax.bar(range(0, len(values)*2, 2), values, width=1, align="center", bottom=bottom, label=possible_fields[i], **plot_kwargs)
        bottom += values
    ax.tick_params(axis='x', which='both', labelbottom=False, top=False, bottom=False)
    ax.set_ylabel("Occurences")
    ax.set_title("Triadic pattern frequency @ inter {}".format(triadic_dataset.niter))
    ax.legend()

    for i in range(0, len(values)*2, 2):
        if triangle_only:
            offset_image(i, triangle_names[i//2+3], ax)
        else:
            offset_image(i, triangle_names[i//2], ax)
    
    return ax

def plot_triadic_pattern_phenotype_from_hist(freq_dict_hist, triangle_list, err=None, triangle_only=False, **plot_kwargs):

    def get_image(name):
        path = r"C:\Users\Matthieu\Documents\_Travail\Stages\Stage M1\Workspace\image\triadic_{}.png".format(name)
        im = plt.imread(path)
        return im

    def offset_image(coord, name, ax):
        img = get_image(name)
        im = OffsetImage(img, zoom=0.2)
        im.image.axes = ax
        ab = AnnotationBbox(im, (coord, 0),  xybox=(0., -16.), frameon=False,
                            xycoords='data',  boxcoords="offset points", pad=0)
        ax.add_artist(ab)

    _, ax = plt.subplots(1, 1, figsize=(8, 5))

    bottom = np.zeros(16)
    if triangle_only:
        bottom = bottom[3::]
    for ph in freq_dict_hist.keys():
        if ph != "Number":
            values = freq_dict_hist[ph] / 3
            ax.bar(range(0, values.size*2, 2), values, width=1, align="center", bottom=bottom, label=ph, **plot_kwargs)
            bottom += values

    if not err is None:
        ax.errorbar(range(0, values.size*2, 2), freq_dict_hist["Number"], yerr=err, fmt="k+", ecolor="k")
    ax.tick_params(axis='x', which='both', labelbottom=False, top=False, bottom=False)
    ax.set_ylabel("Occurences")
    ax.set_title("Triadic pattern frequency from histogram")
    ax.legend()

    for i in range(0, len(values)*2, 2):
        if triangle_only:
            offset_image(i, triangle_list[i//2+3], ax)
        else:
            offset_image(i, triangle_list[i//2], ax)
    
    return ax

def plot_phenotype_combination_per_triangle(triangle_id, combination_dt, th=0):
    """Plot for the triangle with `triangle_id` the possible combination of phenotype calculated in
    `combination_dt` with number of occurences above `th`"""
    dtga = combination_dt.aggregate("Number")
    data = dtga.get_item(triangle_id).get_item("Number").get_all_item()
    data_order = sorted(data.items(), key=lambda x: x[1])
    plot_combination = []
    plot_data = []
    for i in range(len(data_order)):
        if data_order[i][1] > th:
            plot_combination.append(data_order[i][0])
            plot_data.append(data_order[i][1])
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(range(len(plot_data)), plot_data)
    ax.set_xticks(range(len(plot_data)))
    ax.set_xticklabels(plot_combination)
    ax.set_title("Phenotype combinations distribution for triangle {0} with threshold {1}".format(triangle_id, th))
    ax.set_ylabel("Occurence")

    filepath =  r"C:\Users\Matthieu\Documents\_Travail\Stages\Stage M1\Workspace\image\triadic_numbered_{}.png".format(triangle_id)
    img = plt.imread(filepath)
    im = OffsetImage(img, zoom=0.6)
    im.image.axes = ax
    ab = AnnotationBbox(im, (0.1, 0.85),  xybox=(0, -4),
                xycoords='axes fraction',  boxcoords="offset points", pad=0.5)
    ax.add_artist(ab)
    return ax