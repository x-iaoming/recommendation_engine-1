from itertools import product
import numpy as np

grid_params = {
    'reaction_temperature': [100.1, 120.2, 43.3],
    'reaction_time': [20.0, 34.0, 56.0],
    'reaction_pH': [4, 5, 6],
    'pre_heat_standing': [10, 20],
    'teflon_pouch': [0, 1],
    'leak': [0, 1],
    'slow_cool': [0, 1],
    'oil_bath': [0, 1]
}


grid_params_descs = list(grid_params.keys())
grid_params_desc_vals = list(grid_params.values())
#grid_params_desc_combos = product(*grid_params_desc_vals)

#print(len([combo for combo in grid_params_desc_combos]))

grid_numpy_vals = [np.array(val) for val in grid_params_desc_vals]
result = np.meshgrid(*grid_numpy_vals, sparse=False, indexing='ij')
print(result[0])
