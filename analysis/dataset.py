"""analysis/dataset.py
Implement the dataset structure"""

from analysis.analysis_init import *
from networkx import from_numpy_array, betweenness_centrality
from analysis.analysis_utils import get_phenotype_table_from_parameters
from analysis.link_measure import individual_asymmetry, compute_xhi_from_matrix, compute_eta_from_xhi

class Dataset:

    def __init__(self, name="Dataset", niter=0):
        self.name = name
        self.size = 0
        self.data = {}
        self.niter = niter
    
    def init_with_network(self, net):
        """Init Dataset with all local values of the network"""
        self.size = net.parameters["Community size"]
        self.niter = net.parameters["Number of interaction"]
        vertices = net.vertices
        link_adjacency = net.get_link_adjacency_matrix()
        kxgraph_link = from_numpy_array(link_adjacency)
        centralities = betweenness_centrality(kxgraph_link)
        out_degrees = np.sum(link_adjacency, axis=1)
        in_degrees = np.sum(link_adjacency, axis=0)
        asymmetries = individual_asymmetry(link_adjacency)
        expect_proba_ajacency_matrix = net.get_global_expect_probability_matrix()
        games = ["PD", "SH", "SD", "HG"]
        for i in range(self.size):
            i_data = {}
            # Graph index
            index = vertices[i].index
            # Phenotype
            i_data["Phenotype"] = vertices[i].phenotype
            # Memory saturation
            i_data["Saturation"] = len(vertices[i].memory) / vertices[i].memory_size
            # average probabilities per game
            for g in games:
                i_data["p" + g] = vertices[i].get_average_probability(g)
            # average expected probabilities per game
            for g in games:
                i_data["pe" + g] = vertices[i].get_average_expect_probability(g)
            # Expected probability
            i_data["peTotal"] = vertices[i].get_average_global_expect_probability()
            # Correlation
            for g in games:
                i_data["Correlation" + g] = vertices[i].get_average_correlation(g)
            # Outdegree
            i_data["Out degree"] = out_degrees[index]
            # Indegree
            i_data["In degree"] = in_degrees[index]
            # Eta
            xhi = compute_xhi_from_matrix(vertices[i].index, expect_proba_ajacency_matrix, net.parameters["Trust threshold"])
            if np.sum(xhi) > 0:
                i_data["Eta"] = compute_eta_from_xhi(xhi)
            else:
                i_data["Eta"] = np.nan
            # Centrality
            i_data["Centrality"] = centralities[index]
            # Asymmetry
            i_data["Asymmetry"] = asymmetries[index]
            # Adding field to dataset data
            self.data[vertices[i].index] = i_data
        return self

    def init_with_matrices(self, adjacency_dict, parameters, niter):
        """Initialize structure for local measurement only with matrices and parameters"""
        self.size = parameters["Community size"]
        self.niter = niter
        link_adjacency = adjacency_dict["link"]
        kxgraph_link = from_numpy_array(link_adjacency)
        centralities = betweenness_centrality(kxgraph_link)
        out_degrees = np.sum(link_adjacency, axis=1)
        in_degrees = np.sum(link_adjacency, axis=0)
        asymmetries = individual_asymmetry(link_adjacency)
        phenotype_table = get_phenotype_table_from_parameters(parameters)
        games = ["PD", "SH", "SD", "HG"]
        for i in range(self.size):
            i_data = {}
            # Phenotype
            i_data["Phenotype"] = phenotype_table[i]
            # average probabilities per game
            for g in games:
                i_data["p" + g] = np.mean(adjacency_dict["p" + g][i])
            # average expected probabilities per game
            for g in games:
                i_data["pe" + g] = np.mean(adjacency_dict["pe" + g][i])
            # Expected probability
            i_data["peTotal"] = np.mean(adjacency_dict["peTotal"][i])
            # Average correlation
            for g in games:
                i_data["Correlation" + g] = np.mean(adjacency_dict["p" + g][i] * adjacency_dict["pe" + g][i])
            # Outdegree
            i_data["Out degree"] = out_degrees[i]
            # Indegree
            i_data["In degree"] = in_degrees[i]
            # Eta
            xhi = compute_xhi_from_matrix(i, adjacency_dict["peTotal"], parameters["Trust threshold"])
            if np.sum(xhi) > 0:
                i_data["Eta"] = compute_eta_from_xhi(xhi)
            else:
                i_data["Eta"] = np.nan
            # Centrality
            i_data["Centrality"] = centralities[i]
            # Asymmetry
            i_data["Asymmetry"] = asymmetries[i]
            # Adding field to dataset data
            self.data[i] = i_data

    def add(self, id, data_dict={}):
        """add field in dataset"""
        if id in self.data:
            print("WARNING: you are replacing already existing `id` be cautious")
        self.data[id] = data_dict
        self.size += 1
    
    def add_field_in_id(self, id, key, value):
        """Add field in the dictionnary stored in id `id`"""
        if not id in self.data:
            self.add_id(id, {key: value})
            return
        self.data[id][key] = value
    
    def modify_field_in_id(self, id, key, new_value, mode="="):
        """Modify the field `field` for id `id` with `new_value
        Possible modes: `"="`, `"+"`, `"-"`, `"*"`, `"/"`"""
        if key not in self.data[id]:
            if mode == "=" or mode == "+":
                self.add_field_in_id(id, key, new_value)
            else:
                raise Exception("Could not use mode {} to create a field. You probably want to modify a field that does not exist.".format(mode))
        if mode == "=":
            self.data[id][key] = new_value
        elif mode == "+":
            self.data[id][key] += new_value
        elif mode == "-":
            self.data[id][key] -= new_value
        elif mode == "*":
            self.data[id][key] *= new_value
        elif mode == "/":
            self.data[id][key] /= new_value

    def get_item(self, id, query="All"):
        """Get item of id `id` in the dataset matching `query` requirment"""
        output = {}
        if type(query) is str:
            if query == "All":
                if "_value" in self.data[id] and len(self.data[id]) == 1:
                    return self.data[id]["_value"]
                output = self.data[id]
                return output
            if query in self.data[id]:
                output[query] = self.data[id][query]
                return output
            raise KeyError("Unknown field must be in {0}".format(self.data[id].keys()))
        
        try:
            iterator = iter(query)
        except TypeError:
            if query == "All":
                output = self.data[id]
                return output
            if query in self.data[id]:
                output[query] = self.data[id][query]
                return output
            raise KeyError("Unknown field must be in {0}".format(self.data[id].keys()))
        
        for field in iterator:
            if field in self.data[id]:
                output[field] = self.data[id][field]
            else:
                raise KeyError("Unknown field must be in {0}".format(self.data[id].keys()))
        return output
    
    def get_mutliple_item(self, id_list, query="All"):
        """Return all entries with ids in `id_list` of the dataset with fields in `query` """
        output = {}
        for id in id_list:
            output[id] = self.get_item(id, query)

    def get_all_item(self, query="All"):
        """Get all entry of the dataset with fields precised in `query`"""
        output = {}
        for id in self.data.keys():
            output[id] = self.get_item(id, query)
        return output

    def aggregate(self, query):
        """Aggregate the data of same type as in `query` for all the dataset"""
        dtg = DatasetGroup("aggregated " + str(self.name))
        data = self.get_all_item(query)
        for id in self.data.keys():
            sub_data = data[id]
            for key, value in sub_data.items():
                if not dtg.is_in_group(key):
                    dt = Dataset(key, self.niter)
                    dt = dtg.add_dataset(dt)
                else:
                    dt = dtg.get_sub(key)
                dt.add(id, {"_value": value})
        return dtg

    def group_by(self, selector):
        """Regroup data in dataset by value of the field `Selector` """
        dtg = DatasetGroup(selector)
        for i in self.data.keys():
            if selector in self.data[i]:
                if not dtg.is_in_group(self.data[i][selector]):
                    dt = Dataset(self.data[i][selector], self.niter)
                    dt = dtg.add_dataset(dt)
                else:
                    dt = dtg.get_sub(self.data[i][selector])
                dt.add(i, self.get_item(i))
        return dtg

    def copy(self):
        """Copy object"""
        copyds = Dataset(self.name, self.niter)
        for i in self.data.keys():
            copyds.add(i, self.get_item(i).copy())
        return copyds
    
    def keys(self):
        """Return possible keys of the structure"""
        return self.data.keys()

    def __str__(self):
        return "Dataset '" + str(self.name) + "' with data: " + 10 * "-" + "\n" + str(self.data)


class DatasetGroup:

    def __init__(self, name):
        self.name = str(name)
        self.size = 0
        self.subs = {}
    
    def add_dataset(self, dataset):
        """Add a Dataset object `dataset` in subs"""
        self.size += 1
        if dataset.name in self.subs:
            print("WARNING: Replacing dataset might be a name problem")
        self.subs[dataset.name] = dataset
        return dataset
    
    def add_datasetgroup(self, name):
        """Add a DatasetGroup object with name `name` in subs"""
        datagroup = DatasetGroup(name)
        self.subs[name] = datagroup
        return datagroup
    
    def get_sub(self, name):
        """Return the sub with name `name` if in subs"""
        if name in self.subs:
            return self.subs[name]
        raise KeyError("The dataset with name {} is not contained in the group".format(name))

    def is_in_group(self, name):
        """Test is subname `name` is in group"""
        return name in self.subs
    
    def get_item(self, query):
        """Get item(s) matching `query`
        if `query` is iterable will return a DatasetGroup with all subs matched"""
        if type(query) is str:
            if self.is_in_group(query):
                return self.get_sub(query)
            raise KeyError("`query` not valid: not a dataset in the collection")

        try:
            iterator = iter(query)
        except TypeError:
            if self.is_in_group(query):
                return self.get_sub(query)
            raise KeyError("`query` not valid: not a dataset in the collection")
        
        output = DatasetGroup(self.name + str(query))
        for field in iterator:
            if self.is_in_group(field):
                output.add_dataset(self.get_sub(field).copy())
            else:
                raise KeyError("Unknown field")
        return output
    
    def get_all_item(self):
        """Get all subs that are in the DatasetGroup"""
        return self
    
    def group_by(self, selector):
        """Group by each content with selector"""
        copy_dtg = self.copy()
        for name, content in copy_dtg.subs.items():
            copy_dtg.subs[name] = content.group_by(selector)
        return copy_dtg
    
    def aggregate(self, query):
        """Aggregate each content with query"""
        copy_dtg = self.copy()
        for name, content in copy_dtg.subs.items():
            copy_dtg.subs[name] = content.aggregate(query)
        return copy_dtg
    
    def copy(self):
        """Deep copy the object"""
        copy_dtg = DatasetGroup(self.name)
        copy_dtg.size = self.size
        for name, content in self.subs.items():
            copy_dtg.subs[name] = content.copy()
        return copy_dtg
    
    def keys(self):
        """Return possible keys of the DatasetGroup"""
        return self.subs.keys()

    def __str__(self):
        string = "DatasetGroup '" + self.name + "' " + 10 * "-" + "\n"
        for name in self.subs.keys():
            string += str(name) + "\n"
            string += str(self.subs[name]) + "\n"
        return string
