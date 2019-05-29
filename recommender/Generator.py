import json
from itertools import product

class Generator:
    def __init__(self,path_triple,path_params):
        self.path_triple = path_triple
        self.path_params = path_params

    def generate(self):
        # triples_data is an array of dictionary for the triples and amounts:
        triples = open(self.path_triple)
        triples_data = json.load(triples)

        # params_grid_data is a dictionary for the parameters:
        params_grid = open(self.path_params)
        params_grid_data = json.load(params_grid)

        # Get a list of parameters values
        list_params = list(params_grid_data.values())

        # Generate a list of all combos for params 
        params_combos = list(product(*list_params))

        all_combos = []
        # Loop over the array of dictionary for the triples and amounts:
        for i in range(len(triples_data)):
            for j in range(len(params_combos)):
                # Attach one set of triples and amounts to all the combos for params
                params_combos[j] = [list(triples_data[i].values()) + list(params_combos[j])]
            # Add the set of triples and amounts and its combos to all the combos
            all_combos += params_combos  

        # Print the number of combos
        print(len(all_combos))
        return all_combos
