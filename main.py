"""main.py
Main file for running simulation"""
from core import RLNetwork
from utils import readable_adjacency

if __name__ == "__main__":
    net = RLNetwork()
    net.init_with_parameters(r'./parameters.yaml')
    net.play()

    readable_adjacency(net.get_link_adjacency_matrix())