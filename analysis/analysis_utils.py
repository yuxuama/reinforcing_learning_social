"""analysis/analysis_utils.py
All utils for analysis"""

import numpy as np

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