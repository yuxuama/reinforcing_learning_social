"""utils.py
All utils methods for the project"""

from yaml import safe_load
import os
import numpy as np

def parse_parameters(yaml_file):
    """Load parameters for the simulation from a yaml file"""
    stream = open(yaml_file, 'r')
    return safe_load(stream)

def set_parameters_in_hdf5(hdf5_object, parameters):
    """Store the parameters in the dict `parameters` in the HDF5 file object (from h5py) `hdf5_object`"""
    return

def get_parameters_from_hdf5(hdf5_file):
    """Return the parameters stored in the `hdf5_file`"""
    pass

def print_parameters(parameters):
    """Print the parameters extracted from a yaml file"""
    for key, values in parameters.items():
        print("..", key, ": ", values)

def list_all_hdf5(dirpath):
    """Return the list of all the hdf5 files in the directory with path `dirpath`"""
    files = [dirpath + f for f in os.listdir(dirpath)]
    h5_files = []
    for i in range(len(files)):
        if files[i].endswith(".h5"):
            h5_files.append(files[i])
    return h5_files

def readable_adjacency(adjacency_matrix=np.ndarray):
    """Gives the adjacency matric in a form usalble in https://graphonline.top/en/create_graph_by_matrix"""
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            print(adjacency_matrix[i, j], end=", ")
        print()