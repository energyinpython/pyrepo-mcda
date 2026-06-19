import os
from typing import Callable

import numpy as np
import pandas as pd

from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda.mcda_methods import Temporal_PROMETHEE_II
from pyrepo_mcda.promethee_preference_functions import preference_linear_function, preference_usual_function, \
    preference_ushape_function, preference_vshape_function, preference_level_function, preference_gaussian_function
from pyrepo_mcda.weighting_methods import equal_weighting

CSV_DELIMITER = ','

class TemporalPrometheeExample:
    key: str
    performances: dict[str, pd.DataFrame]
    types: list[int]
    _weights: list[float | int]
    preferenceFunctions: list[Callable] | None

    def __init__(
            self,
            key: str,
            performance_paths: dict[str, str],
            types: list[int],
            weights: list[float | int] | Callable,
            preference_functions: list[Callable] | None = None,
    ):
        self.key = key
        self.performances = {}
        self.types = types
        self.preferenceFunctions = preference_functions

        for year, path in performance_paths.items():
            df: pd.DataFrame = pd.read_csv(path, delimiter=CSV_DELIMITER, index_col=0, header=0)

            if len(df.columns) != len(types):
                raise ValueError(
                    f"Number of criteria in performance data does not match number of types and weights provided.")

            self.performances[year] = df

        self.weights = weights

        pass

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: list[float | int] | Callable):
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
        ranks = rank_preferences(scores, reverse=True)

        alt_index = all_net_flows_df.index

        # Save individual CSV files
        scores_df = pd.DataFrame({'scores': scores, 'ranks': ranks}, index=alt_index)
        scores_df.index.name = 'Alternatives'
        scores_df.to_csv(os.path.join(self.output_dir, 'scores.csv'), float_format='%.4f')

        G_df = pd.DataFrame({'G': G}, index=alt_index)
        G_df.index.name = 'Alternatives'
        G_df.to_csv(os.path.join(self.output_dir, 'G.csv'), float_format='%.4f')

        dir_df = pd.DataFrame({'dir': dir}, index=alt_index)
        dir_df['dir_arrow'] = dir_df['dir'].map({
            -1: '$\\downarrow$',
            0: '$\\rightarrow$',
            1: '$\\uparrow$',
        })
        dir_df.index.name = 'Alternatives'
        dir_df.to_csv(os.path.join(self.output_dir, 'dir.csv'))

        all_net_flows_df.to_csv(os.path.join(self.output_dir, 'all_net_flows.csv'), float_format='%.4f')

        # Save all merged into a single file
        all_data = all_net_flows_df.copy()
        all_data['G'] = G
        all_data['dir'] = dir
        all_data['dir_arrow'] = dir_df['dir_arrow']
        all_data['scores'] = scores
        all_data['ranks'] = ranks
        all_data.to_csv(os.path.join(self.output_dir, 'all_data.csv'), float_format='%.4f')

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
    NUM_CRITERIA_PERIODS = 3
    study_periods = TemporalPrometheeExample(
        'periods',
        {
            '$t_1$': 'example_v_1_4/periods/periods_1.csv',
            '$t_2$': 'example_v_1_4/periods/periods_2.csv',
            '$t_3$': 'example_v_1_4/periods/periods_3.csv',
        },
        types=[1 for i in range(NUM_CRITERIA_PERIODS)],
        weights=equal_weighting,
        preference_functions=[preference_usual_function for pf in range(NUM_CRITERIA_PERIODS)],
    )
    study_periods.run()

    NUM_CRITERIA_PFS = 3

    pfs = {
        'usual': preference_usual_function,
        'ushape': preference_ushape_function,
        'vshape': preference_vshape_function,
        'level': preference_level_function,
        'linear': preference_linear_function,
        'gaussian': preference_gaussian_function,
    }

    for pf_name, pf_function in pfs.items():
        study_pfs = TemporalPrometheeExample(
            f"pfs_{pf_name}",
            {
                '$t_1$': 'example_v_1_4/pfs/pfs.csv',
                '$t_2$': 'example_v_1_4/pfs/pfs.csv',
                '$t_3$': 'example_v_1_4/pfs/pfs.csv',
            },
            types=[1 for i in range(NUM_CRITERIA_PFS)],
            weights=equal_weighting,
            preference_functions=[pf_function for pf in range(NUM_CRITERIA_PFS)],
        )
        study_pfs.run()

    NUM_CRITERIA_SHARES = 9
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
        types=[1 for i in range(NUM_CRITERIA_SHARES)],
        weights=equal_weighting,
        preference_functions=[preference_linear_function for pf in range(NUM_CRITERIA_SHARES)],
    )

    study.run()
