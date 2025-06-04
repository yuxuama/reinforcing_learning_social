"""main.py
Main file for running simulation"""
from core import RLNetwork
from core_utils import readable_adjacency
from analysis.network_measure import *
from analysis.dataset import Dataset
from analysis.pattern_measure import measure_frequency_diadic_pattern, measure_global_frequency_triadic_pattern
from plot.dataset_plot import plot_hist_by_phenotype, plot_triadic_pattern_phenotype, plot_bar_diadic_pattern
import matplotlib.pyplot as plt

if __name__ == "__main__":
    net = RLNetwork()
    net.init_with_parameters(r'./parameters.yaml')
    net.play()

    print(memory_saturation_rate(net))
    print(individual_asymmetry(net))
    print(global_asymmetry(net))
    #print(net.vertices[22].probabilities)

    local_dt = Dataset('local', net.parameters["Number of interaction"])
    local_dt.init_with_network(net)
    plot_hist_by_phenotype(local_dt, "Asymmetry")
    plt.show()

    diadic_dt = measure_frequency_diadic_pattern(net.get_link_adjacency_matrix(), net.parameters, net.parameters["Number of interaction"])
    plot_bar_diadic_pattern(diadic_dt)
    plt.show()

    triadic_dt = measure_global_frequency_triadic_pattern(net.get_link_adjacency_matrix(), net.parameters, net.parameters["Number of interaction"])
    plot_triadic_pattern_phenotype(triadic_dt, net.parameters)
    plt.show()
