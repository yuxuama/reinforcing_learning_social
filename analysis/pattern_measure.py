"""analysis/pattern.py
Implement all the methods for analyzing the pattern of the network"""

from analysis.analysis_init import *
from analysis.dataset import Dataset, DatasetGroup
from analysis.analysis_utils import get_phenotype_table_from_parameters

def measure_frequency_diadic_pattern(link_adjacency_matrix, parameters, niter):
    """Measure the frequency of each diadic pattern"""
    pattern_freq = Dataset("Diadic", niter)
    possible_phenotype = list(parameters["Strategy distributions"].keys())
    possible_phenotype.append("Number")
    possible_pattern = [".", "->", "<-", "<->"]
    for pattern in possible_pattern:
        pattern_freq.add(pattern, {ph: 0 for ph in possible_phenotype})
    phenotype_table = get_phenotype_table_from_parameters(parameters)
    size = link_adjacency_matrix.shape[0]
    for i in range(size):
        for j in range(i+1, size):
            ph_i = phenotype_table[i]
            ph_j = phenotype_table[j]
            if link_adjacency_matrix[i, j] == link_adjacency_matrix[j, i]:
                if link_adjacency_matrix[i, j] > 0:
                    pattern_freq.modify_field_in_id("<->", ph_i, 1, mode="+")
                    pattern_freq.modify_field_in_id("<->", ph_j, 1, mode="+")
                    pattern_freq.modify_field_in_id("<->", "Number", 1, mode="+")
                else:
                    pattern_freq.modify_field_in_id(".", ph_i, 1, mode="+")
                    pattern_freq.modify_field_in_id(".", ph_j, 1, mode="+")
                    pattern_freq.modify_field_in_id(".", "Number", 1, mode="+")
            else:
                if link_adjacency_matrix[i, j] > 0:
                    pattern_freq.modify_field_in_id("->", ph_i, 1, mode="+")
                    pattern_freq.modify_field_in_id("<-", ph_j, 1, mode="+")
                    pattern_freq.modify_field_in_id("<-", "Number", 1, mode="+")
                    pattern_freq.modify_field_in_id("->", "Number", 1, mode="+")
                else:
                    pattern_freq.modify_field_in_id("<-", ph_i, 1, mode="+")
                    pattern_freq.modify_field_in_id("->", ph_j, 1, mode="+")
                    pattern_freq.modify_field_in_id("<-", "Number", 1, mode="+")
                    pattern_freq.modify_field_in_id("->", "Number", 1, mode="+")
    
    return pattern_freq

def compute_diadic_histogram(di_pattern_dtg, parameters):
    freq = {"Number": np.zeros(16)}
    possible_phenotype = parameters["Strategy distributions"].keys()
    for ph in possible_phenotype:
        freq[ph] = np.zeros(16)
    
    data_dtg = di_pattern_dtg.aggregate(freq.keys())
    for key in freq.keys():
        freq[key] = np.array(list(data_dtg.get_item(key).get_all_item().values()))
    triangle_name = list(data_dtg.get_item("Number").get_all_item().keys())
    return freq, triangle_name

def get_phenotype_code(ph1, ph2):
    """Return the phenotype code of the link between the two agents with
    phenotypes `ph1` and `ph2`"""
    return ph1[0] + ph2[0]

def revert_phenotype_code(ph_code):
    """Reverse the phenotype code"""
    return ph_code[1] + ph_code[0]

def measure_diadic_pattern_combination(link_adjacency_matrix, phenotype_table, niter):
    """Measure the frequency of each combination among diadic pattern '.', '<->', '->', '<-'"""
    dtg = DatasetGroup('diadic pattern')
    pattern = [".", "<->", "->", "<-"]
    for p in pattern:
        dtg.add_dataset(Dataset(p))
    size = link_adjacency_matrix.shape[0]
    for i in range(size):
        for j in range(i+1, size):
            ph_i = phenotype_table[i]
            ph_j = phenotype_table[j]
            ph_code = get_phenotype_code(ph_i, ph_j)
            r_ph_code = revert_phenotype_code(ph_code)
            if link_adjacency_matrix[i, j] == link_adjacency_matrix[j, i]:
                if link_adjacency_matrix[i, j] > 0:
                    dt = dtg.get_item("<->")
                else:
                    dt = dtg.get_item(".")
                if r_ph_code in dt.data:
                    dt.modify_field_in_id(r_ph_code, "Number", 1, mode="+")
                else:
                    if not ph_code in dt.data: 
                        dt.add(ph_code, {"Number": 0})
                    dt.modify_field_in_id(ph_code, "Number", 1, mode="+")                        
            else:
                if link_adjacency_matrix[i, j] > 0:
                    dt = dtg.get_item("->")
                    if not ph_code in dt.data:
                        dt.add(ph_code, {"Number": 0})
                    dt.modify_field_in_id(ph_code, "Number", 1, mode="+")
                    dt = dtg.get_item("<-")
                    if not r_ph_code in dt.data:
                        dt.add(r_ph_code, {"Number": 0})
                    dt.modify_field_in_id(r_ph_code, "Number", 1, mode="+")
                else:
                    dt = dtg.get_item("<-")
                    if not ph_code in dt.data:
                        dt.add(ph_code, {"Number": 0})
                    dt.modify_field_in_id(ph_code, "Number", 1, mode="+")
                    dt = dtg.get_item("->")
                    if not r_ph_code in dt.data:
                        dt.add(r_ph_code, {"Number": 0})
                    dt.modify_field_in_id(r_ph_code, "Number", 1, mode="+")
    return dtg

triadic_skeleton_global = {
    "000000": {"Number": 0, "Type": "Diadic", "Transitive": False},
    "001010": {"Number": 0, "Type": "Diadic", "Transitive": False},
    "011011": {"Number": 0, "Type": "Diadic", "Transitive": False},
    "002110": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "110002": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "011101": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "012210": {"Number": 0, "Type": "Triadic", "Transitive": True},
    "111111": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "012111": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "111012": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "112121": {"Number": 0, "Type": "Triadic", "Transitive": True},
    "022211": {"Number": 0, "Type": "Triadic", "Transitive": True},
    "211022": {"Number": 0, "Type": "Triadic", "Transitive": True},
    "112112": {"Number": 0, "Type": "Triadic", "Transitive": False},
    "122212": {"Number": 0, "Type": "Triadic", "Transitive": True},
    "222222": {"Number": 0, "Type": "Triadic", "Transitive": True}
}

def is_transitive(triangle_id):
    """Return whether or not the triadic pattern is transitive"""
    return triadic_skeleton_global[triangle_id]["Transitive"]

def complex_degree(i, j, k, link_adjacency):
    """Return complex degree of `i` in the triangle drawn by (i,j,k)"""
    out_deg = link_adjacency[i, j] + link_adjacency[i, k]
    in_deg = link_adjacency[j, i] + link_adjacency[k, i]
    return out_deg, in_deg

def triadic_order(tuple):
    """For recognizing triadic pattern using tuple sort"""
    return tuple[0] + tuple[1], tuple[0]

def get_id_from_degrees(deg_seq):
    """Get the id of the triadic pattern"""
    id = ""
    for i in range(2):
        for j in range(3):
            id += str(deg_seq[j][i])
    return id

def get_phenotype_sequence(deg_seq):
    """Return the order"""
    seq = ""
    for i in range(3):
        seq += deg_seq[i][2][0]
    return seq

def swap(olist, i, j):
    result = olist.copy()
    result[i], result[j] = result[j], result[i]
    return result

def rotate_one(olist):
    result = olist.copy()
    return [result.pop()] + result

def measure_global_frequency_triadic_pattern(link_adjacency_matrix, parameters, niter):
    """Measure the frequency of each triadic pattern and the proportion of each phenotype in the number
    of triangles of a certain type"""
    pattern_freq = Dataset("Triadic", niter)
    phenotype_table = get_phenotype_table_from_parameters(parameters)
    possible_phenotype = parameters["Strategy distributions"].keys()
    for key, dt_dict in triadic_skeleton_global.items():
        pattern_freq.add(key, dt_dict.copy())
        for ph in possible_phenotype:
            pattern_freq.add_field_in_id(key, ph, 0)
    size = link_adjacency_matrix.shape[0]
    for i in range(size):
        for j in range(i+1, size):
            for k in range(j+1, size):
                deg_seq = []
                deg_seq.append(complex_degree(i, j, k, link_adjacency_matrix))
                deg_seq.append(complex_degree(j, i, k, link_adjacency_matrix))
                deg_seq.append(complex_degree(k, j, i, link_adjacency_matrix))
                triangle_id = get_id_from_degrees(sorted(deg_seq, key=triadic_order))

                ph_i = phenotype_table[i]
                ph_j = phenotype_table[j]
                ph_k = phenotype_table[k]
                pattern_freq.modify_field_in_id(triangle_id, "Number", 1, mode="+")
                pattern_freq.modify_field_in_id(triangle_id, ph_i, 1, mode="+")
                pattern_freq.modify_field_in_id(triangle_id, ph_j, 1, mode="+")
                pattern_freq.modify_field_in_id(triangle_id, ph_k, 1, mode="+")
    return pattern_freq

def measure_triadic_pattern_phenotype_combination(link_adjacency_matrix, parameters, niter):
    """Measure the triadic pattern phenotype """
    dtg = DatasetGroup("count")
    for key in triadic_skeleton_global.keys():
        dtg.add_dataset(Dataset(key, niter))
    phenotype_table = get_phenotype_table_from_parameters(parameters)
    size = link_adjacency_matrix.shape[0]
    for i in range(size):
        for j in range(i+1, size):
            for k in range(j+1, size):
                deg_seq = []
                a, b = complex_degree(i, j, k, link_adjacency_matrix)
                deg_seq.append((a, b, phenotype_table[i]))
                a, b = complex_degree(j, i, k, link_adjacency_matrix)
                deg_seq.append((a, b, phenotype_table[j]))
                a, b = complex_degree(k, j, i, link_adjacency_matrix)
                deg_seq.append((a, b, phenotype_table[k]))
                deg_seq = sorted(deg_seq, key=triadic_order)
                triangle_id = get_id_from_degrees(deg_seq)

                dt = dtg.get_item(triangle_id)
                if triangle_id == "222222" or triangle_id == "000000":
                    deg_seq = sorted(deg_seq, key=lambda x: x[2])
                    detected = get_phenotype_sequence(deg_seq)
                elif triangle_id == "111111":
                    ph_one = rotate_one(deg_seq)
                    ph_two = get_phenotype_sequence(rotate_one(ph_one))
                    ph_one = get_phenotype_sequence(ph_one)
                    if ph_one in dt.data:
                        detected = ph_one
                    elif ph_two in dt.data:
                        detected = ph_two
                    else:
                        detected = get_phenotype_sequence(deg_seq)
                    
                else:
                    swap_first = swap(deg_seq, 0, 1)
                    swap_last = swap(deg_seq, 1, 2)
                    if triangle_id == get_id_from_degrees(swap_first):
                        if get_phenotype_sequence(swap_first) in dt.data:
                            detected = get_phenotype_sequence(swap_first)
                    elif triangle_id == get_id_from_degrees(swap_last):
                        if get_phenotype_sequence(swap_last) in dt.data:
                            detected = get_phenotype_sequence(swap_last)
                    else:
                        detected = get_phenotype_sequence(deg_seq)
                
                if detected in dt.data:
                    dt.modify_field_in_id(detected, "Number", 1, mode="+")
                else:
                    dt.add(detected, {"Number": 1})
    return dtg

def compute_triadic_histogram(triadic_pattern_dtg, parameters):
    """Compute the frequency of each motives based on `triadic_pattern_dtg` the DatasetGroup
    of triangle frequencies
    Return the frequency dict of each phenotype and the triangle name list associated"""
    freq = {"Number": np.zeros(16)}
    possible_phenotype = parameters["Strategy distributions"].keys()
    for ph in possible_phenotype:
        freq[ph] = np.zeros(16)
    
    data_dtg = triadic_pattern_dtg.aggregate(freq.keys())
    for key in freq.keys():
        freq[key] = np.array(list(data_dtg.get_item(key).get_all_item().values()))
    triangle_name = list(data_dtg.get_item("Number").get_all_item().keys())
    return freq, triangle_name

def measure_transitivity_from_hist(triadic_hist):
    """Return the value of transitivity of the network from the histogram of frequences"""
    number = triadic_hist["Number"]
    transitive_index = [6, 10, 11, 12, 14, 15]
    transitive = np.sum(number[transitive_index])
    return transitive / np.sum(number)

def measure_transitivity_from_dtg(triadic_pattern_dtg):
    """Return the value of transitivity of the network from the DatasetGroup"""
    dtgga = triadic_pattern_dtg.group_by("Transitive").aggregate("Number")
    number_of_transitive = np.sum(list(dtgga.get_item(True).get_item("Number").get_all_item().values()))
    number_of_non_transitive = np.sum(list(dtgga.get_item(False).get_item("Number").get_all_item().values()))
    return number_of_transitive / (number_of_transitive + number_of_non_transitive)