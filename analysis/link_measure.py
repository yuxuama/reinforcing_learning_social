"""analysis/link_measure.py
Implement all measures on link structure based only on parameters and the link adjacency matrix"""

from analysis.analysis_init import *

def individual_asymmetry(link_adjacency_matrix, median=False):
    n = link_adjacency_matrix.shape[0]
    asymmetries = np.zeros(n)
    number_of_link = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if link_adjacency_matrix[i, j] != link_adjacency_matrix[j, i]:
                asymmetries[i] += 1
                asymmetries[j] += 1
            if link_adjacency_matrix[i, j] > 0 or link_adjacency_matrix[j, i] > 0:
                number_of_link[i] += 1
                number_of_link[j] += 1
    
    for i in range(n):
        if number_of_link[i] < 0:
            asymmetries[i] = np.nan
        else:
            asymmetries[i] = asymmetries[i] / number_of_link[i]

    if median:
        return np.median(asymmetries)

    return asymmetries

def global_asymmetry(link_adjacency_matrix):
    n = link_adjacency_matrix.shape[0]
    total_link = n * (n - 1) / 2
    asymmetric_link = 0
    for i in range(n):
        for j in range(i+1, n):
            if link_adjacency_matrix[i, j] != link_adjacency_matrix[j, i]:
                asymmetric_link += 1
    
    return asymmetric_link / total_link


