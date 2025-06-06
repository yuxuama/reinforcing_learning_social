"""main.py
Main file for running simulation"""
from core import RLNetwork
from analysis.network_measure import *

if __name__ == "__main__":
    net = RLNetwork()
    net.init_with_parameters(r'./parameters.yaml')
    net.play()

    n = net.parameters["Community size"]

    print(memory_saturation_rate(net))
    print(individual_asymmetry(net))
    print(global_asymmetry(net))
    print("Number of interaction per link: ", net.parameters["Number of interaction"] * 2 / (n * (n-1)))
    