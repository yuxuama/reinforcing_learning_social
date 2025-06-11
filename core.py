"""core.py
Implement the network structure
Object RLNetwork and RLVertex defined"""
import numpy as np
import h5py
import os
from core_utils import parse_parameters, print_parameters
from tqdm import tqdm

GAME_TYPE_SIGNATURE = ["PD", "SH", "SD", "HG"]

######################################################################################
# RLNetwork Object
######################################################################################

class RLNetwork:

    def __init__(self, parameters_filename):
        self.seed=None
        self.verbose = False
        self.vertices = []

        # Get parameters
        self.parameters = parse_parameters(parameters_filename)
        if not self.parameters["Output directory"].endswith("/"):
            raise KeyError("The 'Output directory' parameter must end with a '/' as it is a directory")
        if "Verbose" in self.parameters:
            self.verbose = self.parameters["Verbose"]
        if "Seed" in self.parameters:
            self.seed = self.parameters["Seed"]
        # Create vertices
        self.create_all_vertices()

    def reset(self, new_seed=None):
        return self.__init__(new_seed)

    def create_all_vertices(self):
        """Create all vertices with correct phenotype according to parameters"""
        # Phenotypes distribution
        strategy_distrib = self.parameters["Strategy distributions"]
        possible_phenotypes = list(strategy_distrib.keys())
        distribution_grid = [0] # Used to initialized populations of each phenotypes according to parameters.
        for p in strategy_distrib.values():
            distribution_grid.append(distribution_grid[-1] + p)
        distribution_grid = self.parameters["Community size"] * np.array(distribution_grid)

        # Creating vertices
        distrib_pointer = 1
        for i in range(self.parameters["Community size"]):
            if i+1 > distribution_grid[distrib_pointer]:
                distrib_pointer += 1
            v = RLVertex(i, possible_phenotypes[distrib_pointer-1], self.parameters["Memory size"], self.parameters["Initial offset"])
            self.vertices.append(v)

    def generate_name(self, interaction):
        """Generate a name that represent the simulation. This name will be used for saving
        the simulation. The parameters `Seed` can avoid duplicate name.
        name format: '<Distribution>_S<Community size>_T<Trust threshold>_M<Memory size>_N<Number of interaction>(_<seed>)
        distribution format: "Envious: 0.3" -> E3"""
        name = ""
        # distribution
        distribution = ""
        keys = list(self.parameters["Strategy distributions"].keys())
        keys = sorted(keys)
        for key in keys:
            prop = str(self.parameters["Strategy distributions"][key])
            if len(prop) < 2:
                distribution += key[0] + prop
            else:
                distribution += key[0] + prop[2::]
        name += distribution + "_"
        
        # other
        name += "S" + str(self.parameters["Community size"]) + "_"
        name += "T" + str(self.parameters["Trust threshold"]) + "_"
        name += "M" + str(self.parameters["Memory size"]) + "_"
        name += "N" + str(interaction)

        if not self.seed is None:
            name += "_" + str(self.seed)
        
        return name

    def simulate_interaction(self):
        """Simulate an interaction between two randomly taken agents of the network
         We follow the paper and chose a game matrix of the form
        | R | S |
        | T | P |
        where R = 10, P = 5
        and S in [0, 10]
        and T in [5, 15]

        Different conditions for each type of games
        PD: T > R > P > S
        SH: R > T > P > S
        SD: T > R > S > P
        HG: S > P && R > T"""

        # Drawing two different agent
        drawn_agents = np.random.choice(self.vertices, 2, replace=False)
        agent1 = drawn_agents[0]
        agent2 = drawn_agents[1]

        # Drawing game matrix
        drawn_game_type_signature = np.random.choice(GAME_TYPE_SIGNATURE)
        game_matrix = np.array(
            [[10, 0],
             [0, 5]]
        )
        if drawn_game_type_signature == "PD":
            T = np.random.uniform(10, 15)
            S = np.random.uniform(0, 5)
        elif drawn_game_type_signature == "SH":
            T = np.random.uniform(5, 10)
            S = np.random.uniform(0, 5)
        elif drawn_game_type_signature == "SD":
            T = np.random.uniform(10, 15)
            S = np.random.uniform(5, 10)
        elif drawn_game_type_signature == "HG":
            T = np.random.uniform(5, 10)
            S = np.random.uniform(5, 10)
        game_matrix[1, 0] = T
        game_matrix[0, 1] = S

        # Choices of the agent
        agent1_play = agent1.choose(agent2, drawn_game_type_signature)
        agent2_play = agent1.choose(agent1, drawn_game_type_signature)

        # Learning operation for both agents
        agent1.learn(agent2, drawn_game_type_signature, game_matrix, agent1_play, agent2_play)
        agent2.learn(agent1, drawn_game_type_signature, game_matrix, agent2_play, agent1_play)

    def play(self):
        """Run the simulation according to parameters"""
        if self.verbose:
            print_parameters(self.parameters)
        
        if self.parameters["Save mode"] == "Freq":
            to_save_inter = np.linspace(0, self.parameters["Number of interaction"], self.parameters["Number of save"]+1)
            to_save_inter = to_save_inter[1::]
        
        p = 0
        for inter in tqdm(range(self.parameters["Number of interaction"])):
            if self.parameters["Save mode"] == "Freq" and inter == to_save_inter[p]:
                p+=1
                self.save(inter)
            self.simulate_interaction()
        
        if self.parameters["Save mode"] == "Off":
            return
        elif self.parameters["Save mode"] == "Last":
            print("Saving in file: ", self.parameters["Output directory"] + self.generate_name(inter+1) + ".h5")
            self.save()
        elif self.parameters["Save mode"] == "Freq":
            print("Saving in directory: ", self.parameters["Output directory"] + self.generate_name(inter+1) + "/")
            self.save(inter+1)
        else:
            raise KeyError("Unknown save mode: must be in 'Last' or 'Off")

    def save(self, interaction):
        """Saving the current important piece of information of the network in a HDF5 file"""
        name = self.generate_name(interaction)
        if self.parameters["Save mode"] == "Last":
            os.makedirs(self.parameters["Output directory"], exist_ok=True)
            dir_name = ""
        elif self.parameters["Save mode"] == "Freq":
            dir_name = self.generate_name(self.parameters["Number of interaction"]) + "/"
            os.makedirs(self.parameters["Output directory"] + dir_name, exist_ok=True)
        filepath = self.parameters["Output directory"] + dir_name + name + ".h5"
        hdf5_file = h5py.File(filepath, "w")

        # Saving all adjacency matrices
        for g in GAME_TYPE_SIGNATURE:
            hdf5_file["p" + g] = self.get_probability_matrix(g)
            hdf5_file["pe" + g] = self.get_expect_probability_matrix(g)
        hdf5_file["peTotal"] = self.get_global_expect_probability_matrix()
        hdf5_file["link"] = self.get_link_adjacency_matrix()

        # Saving parameters
        pgroup = hdf5_file.create_group("Parameters", track_order=True)
        for key, value in self.parameters.items():
            if key != "Strategy distributions":
                if key == "Number of interaction":
                    pgroup[key] = interaction
                else:
                    pgroup[key] = value
        dgroup = pgroup.create_group("Strategy distributions")
        for ph, proba in self.parameters["Strategy distributions"].items():
            dgroup[ph] = proba

        hdf5_file.close()      

    def get_probability_matrix(self, game_type_signature):
        """Return the adjacency matrix of the probability of agent to cooperate for the game
        `game_type_signature` """
        size = self.parameters["Community size"]
        matrix = np.zeros((size, size), dtype=float)
        for i in range(size):
            for j in range(size):
                if i != j:
                    matrix[i, j] = self.vertices[i].get_probability(self.vertices[j], game_type_signature)
        return matrix

    def get_expect_probability_matrix(self, game_type_signature):
        """Return the adjacency matrix of the expected probability of the other agent to 
        cooperate for the game `game_type_signature` """
        size = self.parameters["Community size"]
        matrix = np.zeros((size, size), dtype=float)
        for i in range(size):
            for j in range(size):
                if i != j:
                    matrix[i, j] = self.vertices[i].get_expect_probability(self.vertices[j], game_type_signature)
        return matrix

    def get_global_expect_probability_matrix(self):
        """Return the adjacency matrix of the expected probability of the other agent to 
        cooperate in general"""
        size = self.parameters["Community size"]
        matrix = np.zeros((size, size), dtype=float)
        for i in range(size):
            for j in range(size):
                if i != j:
                    matrix[i, j] = self.vertices[i].get_global_expect_probability(self.vertices[j])
        return matrix
    
    def get_correlation_adjacency_matrix(self, game_type_signature):
        """Return the adjacency matrix of the correlation between probability and exepected probability"""
        size = self.parameters["Community size"]
        corr_adjacency = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                corr_adjacency[i, j] = self.vertices[i].get_correlation(self.vertices[j], game_type_signature)
        return corr_adjacency

    def get_link_adjacency_matrix(self):
        """Return the link adjacency matrix based on the value of "Trust threshold" and the 
        expected probabilities of each vertices"""
        size = self.parameters["Community size"]
        link_adjacency_matrix = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if i != j:
                    p = self.vertices[i].get_global_expect_probability(self.vertices[j])
                    if p >= self.parameters["Trust threshold"]: 
                        link_adjacency_matrix[i, j] = 1
        
        return link_adjacency_matrix
    
    def get_edge_list(self):
        """Generate the edge list of the network having the layer of expected probability and link matrix
        This is for the conversion to `graph-tool` module graph structure"""
        global_expect_probability_matrix = self.get_global_expect_probability_matrix()
        link_matrix = self.get_link_adjacency_matrix()
        edge_list = []
        size = global_expect_probability_matrix.shape[0]
        for i in range(size):
            for j in range(size):
                edge_list.append((i, j, global_expect_probability_matrix[i, j], link_matrix[i, j]))
        return edge_list

    def get_phenotype_table(self):
        """Return the correspondance table index - phenotype"""
        ph_table = []
        for i in range(self.parameters["Community size"]):
            ph_table.append(self.vertices[i].phenotype)
        return ph_table

######################################################################################
# RLVertex object
######################################################################################

INITIAL_PROBABILITIES = {
    "Envious": {
        "PD": {'self': 0, "other C": 0, "other total": 0},
        "SH": {'self': 0, "other C": 0, "other total": 0},
        "SD": {'self': 0, "other C": 0, "other total": 0},
        "HG": {'self': 1, "other C": 0, "other total": 0},
    },
    "Optimist": {
        "PD": {'self': 0, "other C": 0, "other total": 0},
        "SH": {'self': 1, "other C": 0, "other total": 0},
        "SD": {'self': 0, "other C": 0, "other total": 0},
        "HG": {'self': 1, "other C": 0, "other total": 0},
    },
    "Pessimist": {
        "PD": {'self': 0, "other C": 0, "other total": 0},
        "SH": {'self': 0, "other C": 0, "other total": 0},
        "SD": {'self': 1, "other C": 0, "other total": 0},
        "HG": {'self': 1, "other C": 0, "other total": 0},
    },
    "Random": {
        "PD": {'self': 0.5, "other C": 0, "other total": 0},
        "SH": {'self': 0.5, "other C": 0, "other total": 0},
        "SD": {'self': 0.5, "other C": 0, "other total": 0},
        "HG": {'self': 0.5, "other C": 0, "other total": 0},
    },
    "Trustful": {
        "PD": {'self': 1, "other C": 0, "other total": 0},
        "SH": {'self': 1, "other C": 0, "other total": 0},
        "SD": {'self': 1, "other C": 0, "other total": 0},
        "HG": {'self': 1, "other C": 0, "other total": 0},
    }
}

def get_posterior_expected_probability(Nc, Ntot, phenotype):
    """Return the probability of the other to cooperate knowing the number of time it cooperated
    in the past and the total number of interaction
    For now we used Bayesian inference assuming a prior P(p_expect = x) = x * (1 - x).
    This is roughly equivalent to considering that C and D are equiprobable and doing an entropy analysis.
    In this analysis we can get the distribution of P(p_expect) with respect to a certain number of "memory"
    tries `N_mem`. We are essentially asking how probable it is to have P(p_expect=x) based on previously 
    experienced situation of N_mem successive draws (C or D with 50-50 chance).
    A prior x * (1 - x) is roughly equivalent to N_mem = 3.
    N_mem represent how strongly one believe the other will behave randomly
    """
    #if phenotype == "Trustful" or phenotype == "Optimist":
    #    return (Nc + 1) / (Ntot + 1)
    #elif phenotype == "Pessimist":
    #    return Nc / (Ntot + 1)
    
    return (Nc + 1) / (Ntot + 2)

class RLVertex:

    def __init__(self, index, phenotype, memory_size, initial_offset):
        """Initializing object properties"""
        self.index = index # Must be unique to this object (is used as hash key)
        self.phenotype = phenotype
        self.memory_size = memory_size
        self.offset = initial_offset

        self.memory = []
        self.probabilities = {}
    
    def choose(self, other, game_type_signature):
        """Given an interaction matrix and the probabilities for `other` select a play
        at random
        Return 0 for cooperation and 1 for defection"""
        if other not in self.probabilities:
            self.probabilities[other] = {}
            for game_type, game_dict in INITIAL_PROBABILITIES[self.phenotype].items():
                self.probabilities[other][game_type] = {}
                for key, proba in game_dict.items():
                    if key == "self":
                        if proba == 0:
                            proba = self.offset
                        elif proba == 1:
                            proba = 1 - self.offset
                    self.probabilities[other][game_type][key] = proba
        
        p = self.get_probability(other, game_type_signature)
        draw = np.random.random()
        if draw < p:
            return 0
        return 1

    def learn(self, other, game_type_signature, game_matrix, self_play, other_play):
        """Compute the inspiration level based on probabilities and update them according
        to the result of the simulation"""
        p = self.get_probability(other, game_type_signature)
        o_p = self.get_expect_probability(other, game_type_signature)

        inspiration = np.array([p, 1-p]) @ game_matrix @ np.array([o_p, 1-o_p])
        payoff = game_matrix[self_play, other_play]
        s = (payoff - inspiration) / np.max(game_matrix) 

        if s >= 0:
            self.set_probability(other, game_type_signature, p + s*(1-p))
        else:
            self.set_probability(other, game_type_signature, p + s*p)
        
        self.update_memory(other, game_type_signature, other_play)

    def update_memory(self, other, game_type_signature, other_play):
        """Update memory. In particular keeps it below the "Memory size" parameter of the simulation"""
        self.memory.append((other, game_type_signature, other_play))
        self.probabilities[other][game_type_signature]["other total"] += 1
        if other_play == 0: # If it was a cooperation
            self.probabilities[other][game_type_signature]["other C"] += 1

        if len(self.memory) > self.memory_size:
            forgot_other, forgot_gametype, forgot_play = self.memory.pop(0)
            self.probabilities[forgot_other][forgot_gametype]["other total"] -= 1
            if forgot_play == 0: # If it was a cooperation
                self.probabilities[forgot_other][forgot_gametype]["other C"] -= 1

    def get_probability(self, other, game_type_signature):
        """Return the probability of playing 'C' (cooperation) with the agent `other` playing to 
        the game indentified by `game_type_signature`"""
        if other in self.probabilities:
            return self.probabilities[other][game_type_signature]["self"]
        else:
            return INITIAL_PROBABILITIES[self.phenotype][game_type_signature]["self"]
    
    def get_average_probability(self, game_type_signature):
        """Return the average probability of the agent to cooperate for the game indicated 
        by `game_type_signature`. The average is computed only for people with who it has 
        interacted in the past."""
        probas = []
        for other in self.probabilities:
            probas.append(self.get_probability(other, game_type_signature))
        return np.mean(probas)
    
    def get_expect_probability(self, other, game_type_signature):
        """Return, from the perspective of the agent, the probability for the agent `other`
        to cooperate for the type of game identified by `game_type_signature`"""
        if other in self.probabilities:
            Nc = self.probabilities[other][game_type_signature]["other C"]
            Ntot = self.probabilities[other][game_type_signature]["other total"]
            return get_posterior_expected_probability(Nc, Ntot, self.phenotype)

        return 0.5
    
    def get_average_expect_probability(self, game_type_signature):
        """Return the average probability of the agent to cooperate for the game indicated 
        by `game_type_signature`. The average is computed only for people with who it has 
        interacted in the past."""
        probas = []
        for other in self.probabilities:
            probas.append(self.get_expect_probability(other, game_type_signature))
        return np.mean(probas)

    def get_global_expect_probability(self, other):
        """Return, from the perspective of the agent, the probability for the agent `other`
        to cooperate in general"""
        if other in self.probabilities:
            Nc = 0
            Ntot = 0
            for game_type in GAME_TYPE_SIGNATURE:
                Nc += self.probabilities[other][game_type]["other C"]
                Ntot += self.probabilities[other][game_type]["other total"]
            return get_posterior_expected_probability(Nc, Ntot, self.phenotype)
        
        return 0.5
    
    def get_average_global_expect_probability(self):
        probas = []
        for other in self.probabilities:
            probas.append(self.get_global_expect_probability(other))
        return np.mean(probas)
    
    def get_correlation(self, other, game_type_signature):
        """Return the correlation value between the probability of cooperating and the probability with which
        the agent expect the other to cooperate"""
        return self.get_probability(other, game_type_signature) * self.get_expect_probability(other, game_type_signature)

    def get_average_correlation(self, game_type_signature):
        """Return the correlation between the probability of cooperating and the expected probability of
        another cooperating"""
        correlation = []
        for other in self.probabilities:
            corr = self.get_correlation(other, game_type_signature)
            correlation.append(corr)
        return np.mean(corr)
    
    def set_probability(self, other, game_type_signature, value):
        """Set the probability of cooperating in the game identified by `game_type_signature` to `value`"""
        if not other in self.probabilities:
            self.probabilities[other] = {}
            for game_type, game_dict in INITIAL_PROBABILITIES[self.phenotype].items():
                self.probabilities[other][game_type] = {}
                for key, proba in game_dict.items():
                    self.probabilities[other][game_type][key] = proba
        
        self.probabilities[other][game_type_signature]["self"] = value

    def __hash__(self):
        return self.index
    
    def __str__(self):
        return str(self.index)