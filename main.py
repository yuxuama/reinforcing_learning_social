"""main.py
Main file for running simulation"""
from core import RLNetwork

if __name__ == "__main__":
    net = RLNetwork(r'./parameters.yaml')
    net.play()