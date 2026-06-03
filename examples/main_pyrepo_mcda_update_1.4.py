import os
from typing import Callable

import numpy as np
import pandas as pd

from pyrepo_mcda.mcda_methods import Temporal_PROMETHEE_II
from pyrepo_mcda.promethee_preference_functions import preference_linear_function
from pyrepo_mcda.weighting_methods import equal_weighting

CSV_DELIMITER = ','
FORMAT = 'png'

class TemporalPrometheeExample:
    key: str
    performances: dict[str, pd.DataFrame]
    types: list[int]
    _weights: list[float|int]
    preferenceFunctions: list[Callable] | None

    def __init__(
            self,
            key: str,
            performance_paths: dict[str, str],
            types: list[int],
            weights: list[float|int] | Callable,
            preference_functions: list[Callable] | None = None,
    ):
        self.key = key
        self.performances = {}
        self.types = types
        self.preferenceFunctions = preference_functions

        for year, path in performance_paths.items():
            df: pd.DataFrame = pd.read_csv(path, delimiter=CSV_DELIMITER, index_col=0, header=0)

            if len(df.columns) != len(types):
                raise ValueError(f"Number of criteria in performance data does not match number of types and weights provided.")

            self.performances[year] = df

        self.weights = weights

        pass

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: list[float|int] | Callable):
        if isinstance(weights, list):
            self._weights = weights
        elif isinstance(weights, Callable):
            if len(self.matrices) == 0:
                raise ValueError("No performance matrices loaded. Cannot compute weights.")

            matrix = next(iter(self.matrices.values()))
            computed = weights(matrix)
            self._weights = computed

    @property
    def output_dir(self):
        return f"results_update_1.4/{self.key}"

    @property
    def matrices(self) -> dict[str, np.ndarray]:
        return {year: df.to_numpy() for year, df in self.performances.items()}

    def run(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        tp = Temporal_PROMETHEE_II()
        scores, (G, dir, all_net_flows_df) = tp(
            matrices=self.matrices,
            weights=self.weights_np,
            types=self.types_np,
            preference_functions=self.preferenceFunctions,
            alt_names=self.alternatives,
        )

        pass

    @property
    def weights_np(self) -> np.ndarray:
        return np.array(self._weights)

    @property
    def types_np(self) -> np.ndarray:
        return np.array(self.types)

    @property
    def alternatives(self) -> list[str]:
        if len(self.performances) == 0:
            return []
        else:
            return list(next(iter(self.performances.values())).index)

if __name__ == '__main__':
    NUM_CRITERIA = 9
    study = TemporalPrometheeExample(
        'shares',
        {
            '2013': 'example_v_1_4/shares/shares_2013.csv',
            '2014': 'example_v_1_4/shares/shares_2014.csv',
            '2015': 'example_v_1_4/shares/shares_2015.csv',
            '2016': 'example_v_1_4/shares/shares_2016.csv',
            '2017': 'example_v_1_4/shares/shares_2017.csv',
            '2018': 'example_v_1_4/shares/shares_2018.csv',
            '2019': 'example_v_1_4/shares/shares_2019.csv',
            '2020': 'example_v_1_4/shares/shares_2020.csv',
        },
        types=[1 for i in range(NUM_CRITERIA)],
        weights=equal_weighting,
        preference_functions=[preference_linear_function for pf in range(NUM_CRITERIA)],
    )

    study.run()
