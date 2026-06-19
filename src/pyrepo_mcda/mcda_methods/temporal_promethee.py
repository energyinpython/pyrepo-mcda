from typing import Callable

import numpy as np
import pandas as pd

from pyrepo_mcda.mcda_methods import PROMETHEE_II, DARIA


class Temporal_PROMETHEE_II:

    def __init__(self):
        """
        Create the Temporal PROMETHEE II method object
        """
        pass

    def __call__(
            self,
            matrices: dict[str, np.ndarray],
            weights: np.ndarray,
            types: np.ndarray,
            preference_functions: list[Callable] | None = None,
            ps: list[float|int] | None = None,
            qs: list[float|int] | None = None,
            alt_names: list[str]|None = None
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
        """
        Calculate Temporal PROMETHEE II scores and helper outputs.

        Returns
        -------
        tuple[np.ndarray, tuple[np.ndarray, np.ndarray, pd.DataFrame]]
            final_S : Updated final efficiencies.
            tuple[np.ndarray, np.ndarray, pd.DataFrame]
                G : Standard deviation-based variability for each alternative.
                dir : Variability direction for each alternative.
                all_net_flows_df : DataFrame with PROMETHEE II net flows per year.
        """
        promethee_II = PROMETHEE_II()

        all_net_flows = {}

        for year, matrix in matrices.items():
            net_flows = promethee_II(matrix, weights, types, preference_functions, ps, qs)
            all_net_flows[year] = net_flows

        all_net_flows_df = pd.DataFrame(all_net_flows)
        if alt_names:
            all_net_flows_df.index = alt_names

        all_net_flows_df.rename_axis('Alternatives', inplace=True)


        # perform DARIA calculations

        # rows: years, columns: alternatives
        matrix = all_net_flows_df.T.to_numpy()

        # PROMETHEE II orders preferences in descending order
        type = 1

        # Calculate efficiencies variability using methods from DARIA class
        # Create the DARIA class object
        daria = DARIA()
        # Calculate variability values for each alternative with Standard deviation using the method from DARIA class
        G = daria._std(matrix)
        # Calculate variability directions for each alternative using the method from DARIA class
        _, dir = daria._direction(matrix, type)

        # The most recent year will be updated by variability
        S = all_net_flows_df[all_net_flows_df.columns[-1]].to_numpy()

        # update efficiencies using the method from DARIA class
        final_S = daria._update_efficiency(S, G, dir)

        return final_S, (G, dir, all_net_flows_df)
