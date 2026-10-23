import os
import numpy as np
import pandas as pd

from pyrepo_mcda.mcda_methods import SAW, TOPSIS, VIKOR, WASPAS, PROMETHEE_II
from pyrepo_mcda.normalizations import minmax_normalization
from pyrepo_mcda.promethee_preference_functions import preference_linear_function


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(base_dir, 'example_v_1_5', 'epi2026.csv'))

    criteria = ['HLT', 'ECO', 'PCC']
    matrix = data[criteria].to_numpy(dtype=float)
    weights = np.array([0.45, 0.25, 0.3])
    types = np.ones(len(criteria))

    saw = SAW(normalization_method=minmax_normalization)
    topsis = TOPSIS()
    vikor = VIKOR(v = 0.5)
    waspas = WASPAS(lambda_param = 0.5, normalization_method=minmax_normalization)
    promethee = PROMETHEE_II()

    results = pd.DataFrame(index = pd.Index(data['ISO'], name = 'ISO'))
    results['SAW'] = saw(matrix, weights, types)
    results['TOPSIS'] = topsis(matrix, weights, types)
    results['VIKOR'] = vikor(matrix, weights, types)
    results['WASPAS'] = waspas(matrix, weights, types)
    stdev = np.round(np.std(matrix, axis = 0), 2)
    pf = [preference_linear_function] * len(criteria)
    results['PROMETHEE'] = promethee(matrix, weights, types, preference_functions = pf, p = 1 * stdev, q = 0.5 * stdev)
    results = results.round(10)

    print(results)

    out_dir = os.path.join(base_dir, 'results_update_1.5')
    os.makedirs(out_dir, exist_ok = True)
    results.to_csv(os.path.join(out_dir, 'scores.csv'))


if __name__ == '__main__':
    main()
