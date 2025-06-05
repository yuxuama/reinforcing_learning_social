"""main.py
Main file for running simulation"""
from core import RLNetwork
from core_utils import readable_adjacency
from analysis.network_measure import *
from analysis.dataset import Dataset
from analysis.link_measure import mean_histograms
from analysis.pattern_measure import measure_frequency_diadic_pattern, measure_global_frequency_triadic_pattern
from plot.dataset_plot import plot_hist_by_phenotype, plot_triadic_pattern_phenotype, plot_bar_diadic_pattern
from plot.plot import plot_histogram_per_phenotype
import matplotlib.pyplot as plt

if __name__ == "__main__":
    net = RLNetwork()
    net.init_with_parameters(r'./parameters.yaml')
    net.play()

    n = net.parameters["Community size"]

    print(memory_saturation_rate(net))
    print(individual_asymmetry(net))
    print(global_asymmetry(net))
    print("Number of interaction per link: ", net.parameters["Number of interaction"] * 2 / (n * (n-1)))
    #print(net.vertices[22].probabilities)

    mean_hist, bins = mean_histograms(net.get_global_expect_probability_matrix(), net.parameters["Trust threshold"], net.get_phenotype_table(), bin_number=10)
    plot_histogram_per_phenotype(mean_hist, bins)
    plt.savefig(r"./figure/trust.png")

    local_dt = Dataset('local', net.parameters["Number of interaction"])
    local_dt.init_with_network(net)
    plot_hist_by_phenotype(local_dt, "Asymmetry")
    plt.savefig(r"./figure/asymmetry.png")

    to_plot = ["pPD", "pSH", "pSD", "pHG", "pePD", "peSH", "peSD", "peHG", "CorrelationPD", "CorrelationSH", "CorrelationSD", "CorrelationHG"]
    for quantity in to_plot:
        plot_hist_by_phenotype(local_dt, quantity)
        plt.savefig(r"./figure/{}.png".format(quantity))
    
    diadic_dt = measure_frequency_diadic_pattern(net.get_link_adjacency_matrix(), net.parameters, net.parameters["Number of interaction"])
    plot_bar_diadic_pattern(diadic_dt)
    plt.savefig(r"./figure/diadic.png")

    triadic_dt = measure_global_frequency_triadic_pattern(net.get_link_adjacency_matrix(), net.parameters, net.parameters["Number of interaction"])
    plot_triadic_pattern_phenotype(triadic_dt, net.parameters)
    plt.savefig(r"./figure/triadic.png")
