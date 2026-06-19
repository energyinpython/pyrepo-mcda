# Illustrative example of Temporal PROMETHEE II method

# import necessary packages
import numpy as np
import pandas as pd

# Import necessary `pyrepo_mcda` Package modules.
from pyrepo_mcda.mcda_methods import Temporal_PROMETHEE_II
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda.weighting_methods import equal_weighting
from pyrepo_mcda.promethee_preference_functions import preference_linear_function

# Define decision problem

# Each data matrix consists of 32 alternatives (countries) and 9 criteria (types of alternative fuels).
matrices = {}
for year in range(2013, 2021):
    matrices[f"{year}"] = pd.read_csv(f"example_v_1_4/shares/shares_{year}.csv", delimiter=',', index_col=0, header=0).to_numpy()

# Equal weighting is used for the evaluation of the alternatives.
weights = equal_weighting(matrices['2013'])

# All criteria are considered benefit criteria to be maximized.
types = np.array([1] * len(weights))

# Linear preference function is used for all criteria.
# No preference `p` and indifference `q` thresholds are provided,
# so the default `p = 2 * u` and `q = 0.5 * u` based on standard
# deviation `u` will be used.
preference_functions = [preference_linear_function for pf in range(len(weights))]

# Initialize Temporal PROMETHEE II method object
tp = Temporal_PROMETHEE_II()

# Calculate temporal preference scores
scores, (G, dir, all_net_flows_df) = tp(matrices=matrices, weights=weights, types=types, preference_functions=preference_functions)

# Generate ranking. The best alternative has the highest score.
ranks = rank_preferences(scores, reverse=True)

print('Scores:', np.round(scores, 4))
print('Ranks:', ranks)
