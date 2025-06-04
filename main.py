"""main.py
Main file for running simulation"""
from core import RLNetwork
from core_utils import readable_adjacency
from analysis.network_measure import *

if __name__ == "__main__":
    net = RLNetwork()
    net.init_with_parameters(r'./parameters.yaml')
    net.play()

    print(memory_saturation_rate(net))
    print(individual_asymmetry(net))
    print(global_asymmetry(net))
    #print(net.vertices[22].probabilities)