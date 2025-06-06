"""analysis/link_measure.py
Implement all measures on link structure based only on parameters and the link adjacency matrix"""

from analysis.analysis_init import *
from scipy.optimize import curve_fit

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


# histograms and êtas

def trust_histogram(i, expect_proba_ajacency_matrix, trust_threshold, bin_number=10):
    """Compute the histogram for the vertex with index `i` with `bin_number` bins"""
    data = expect_proba_ajacency_matrix[i]
    selector = data > trust_threshold
    data = data[selector]
    return np.histogram(data, bins=bin_number)

def mean_trust_histograms(expect_proba_ajacency_matrix, trust_threshold, phenotype_table, bin_number=10):
    """Compute the average histogram for each phenotype and the global popupaltion"""
    histograms = {"Global": np.zeros(bin_number)}
    phenotype_numbers = {}
    size = expect_proba_ajacency_matrix.shape[0]
    for i in range(size):
        ph = phenotype_table[i]
        if not ph in histograms:
            histograms[ph] = np.zeros(bin_number)
            phenotype_numbers[ph] = 0
        hist, _ = trust_histogram(i, expect_proba_ajacency_matrix, trust_threshold, bin_number=bin_number)
        histograms[ph] += hist
        histograms["Global"] += hist
        phenotype_numbers[ph] += 1
    
    for key in histograms.keys():
        if key == "Global":
            histograms[key] = histograms[key] / size
        else:
            histograms[key] = histograms[key] / phenotype_numbers[key]
    
    bins = np.linspace(trust_threshold, 1, bin_number+1)
    
    return histograms, bins

def compute_xhi_from_histogram(histogram):
    """Compute the function xhi. It has as many points as in the number of bins of the histogram"""
    L = np.sum(histogram)
    xhi = histogram / L
    xhi = np.flip(xhi)
    for i in range(1, len(xhi)):
        xhi[i] += xhi[i-1]
    return xhi

def compute_xhi_from_matrix(i, expect_proba_ajacency_matrix, trust_threshold, bin_number=10):
    """Compute the function xhi. It has as many points as in the number of specified by `bin_number`"""
    hist, _ = trust_histogram(i , expect_proba_ajacency_matrix, trust_threshold, bin_number=bin_number)
    return compute_xhi_from_histogram(hist)

def compute_mean_xhis(expect_proba_ajacency_matrix, trust_threshold, phenotype_table, bin_number=10):
    """Compute the average xhis per phenotype and for global"""
    mean_xhis = {"Global": np.zeros(bin_number)}
    phenotype_numbers = {}
    size = expect_proba_ajacency_matrix.shape[0]
    for i in range(size):
        ph = phenotype_table[i]
        if not ph in mean_xhis:
            mean_xhis[ph] = np.zeros(bin_number)
            phenotype_numbers[ph] = 0
        hist, _ = trust_histogram(i, expect_proba_ajacency_matrix, trust_threshold, bin_number=bin_number)
        if np.sum(hist) > 0:
            xhi = compute_xhi_from_histogram(hist)
            mean_xhis[ph] += xhi
            mean_xhis["Global"] += xhi
            phenotype_numbers[ph] += 1
    
    for key in mean_xhis.keys():
        if key == "Global":
            mean_xhis[key] = mean_xhis[key] / size
        else:
            mean_xhis[key] = mean_xhis[key] / phenotype_numbers[key]
    
    return mean_xhis

def compute_eta_from_xhi(xhi):
    """Compute the value of eta based on the function `xhi` given in parameters"""
    size = xhi.size
    t = np.linspace(0, 1, size)
    model = lambda j, eta: (np.exp(eta * j) - 1) / (np.exp(eta) - 1)
    popt, _ = curve_fit(model, t, xhi)
    return popt[0]