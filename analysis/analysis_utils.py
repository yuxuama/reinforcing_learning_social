"""analysis/analysis_utils.py
All utils for analysis"""

from analysis.analysis_init import *
import h5py

GAME_TYPE_SIGNATURE = ["PD", "SH", "SD", "HG"]

def get_phenotype_table_from_parameters(parameters):
    """Give the link vertex index - vertex phenotype according to the parameters"""
    strategy_distrib = parameters["Strategy distributions"]
    possible_phenotypes = list(strategy_distrib.keys())
    distribution_grid = [0] # Used to initialized populations of each phenotypes according to parameters.
    for p in strategy_distrib.values():
        distribution_grid.append(distribution_grid[-1] + p)
    distribution_grid = parameters["Community size"] * np.array(distribution_grid)
        
    # Creating vertices
    phenotypes_table = ["" for i in range(parameters["Community size"])]
    d_pointer = 1
    for i in range(parameters["Community size"]):
        if i+1 > distribution_grid[d_pointer]:
            d_pointer += 1
        phenotype = possible_phenotypes[d_pointer - 1]
        phenotypes_table[i] = phenotype
    
    return phenotypes_table

def extract_parameters_from_hdf5(filepath):
    """Extract parameters dict saved in the hdf5 file"""
    parameters = {}
    string_keys = {"Output directory", "Save mode"}
    hdf5_file = h5py.File(filepath)
    grp = hdf5_file["Parameters"]
    for p in grp.keys():
        if p != "Strategy distributions":
            if p in string_keys:
                parameters[p] = grp.get(p)[()].decode("ascii")
            else:
                parameters[p] = grp.get(p)[()]
    subgrp = grp["Strategy distributions"]
    strategy_distrib = {}
    for ph in subgrp.keys():
        strategy_distrib[ph] = subgrp.get(ph)[()]
    parameters["Strategy distributions"] = strategy_distrib
    hdf5_file.close()

    return parameters

def extract_all_info_from_hdf5(filepath):
    """Extract all adjacency matrices and parameters from the hdf5 file"""
    parameters = extract_parameters_from_hdf5(filepath)
    adjacency_matrices = {}
    hdf5_file = h5py.File(filepath)
    for groups in hdf5_file.keys():
        if groups != "Parameters":
            adjacency_matrices[groups] = hdf5_file[groups][:]
    
    hdf5_file.close()

    return adjacency_matrices, parameters

def generate_edge_list_from_matrices(global_expect_probability_matrix, link_matrix):
    """Generate the edge list of the network having the layer of expected probability and link matrix"""
    edge_list = []
    size = global_expect_probability_matrix.shape[0]
    for i in range(size):
        for j in range(size):
            edge_list.append((i, j, global_expect_probability_matrix[i, j], link_matrix[i, j]))
    return edge_list

