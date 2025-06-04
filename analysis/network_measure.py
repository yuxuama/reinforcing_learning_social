"""analysis/network_measure.py
Implement all the measure method that requires a RLNetwork object to be done"""

from analysis.analysis_init import *
import analysis.link_measure

def memory_saturation_rate(net):
    """Measure the proporition of vertices having reached full memory capacity"""
    size = net.parameters["Community size"]
    saturated_number = 0
    for v in net.vertices:
        if len(v.memory) == v.memory_size:
            saturated_number += 1
    
    return saturated_number / size

def individual_asymmetry(net, median=False):
    """Measure per vertex the proportion of asymmetric links"""
    return analysis.link_measure.individual_asymmetry(net.get_link_adjacency_matrix(), median=median)

def global_asymmetry(net):
    """Measure globally the asymmetry rate"""
    return analysis.link_measure.global_asymmetry(net.get_link_adjacency_matrix())