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
            variability_function: Callable[[np.ndarray], np.ndarray] | None = None,
            preference_functions: list[Callable] | None = None,
            ps: list[float | int] | None = None,
            qs: list[float | int] | None = None,
            alt_names: list[str] | None = None
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
        
        """
        Calculate Temporal PROMETHEE II scores considering variability 
        of alternative preferences over multiple time periods.

        The method first calculates PROMETHEE II net flows for each 
        decision matrix representing a separate time period. Next, variability 
        and direction of changes in preferences are determined using selected 
        DARIA measures. Finally, the PROMETHEE II scores from the most recent 
        period are updated using variability information to obtain temporal 
        preference scores. 
        
        Decision matrices must contain the same number of alternatives and criteria
        in all periods.

        Parameters
        ----------
        matrices : dict[str, np.ndarray]
            Dictionary containing decision matrices for subsequent time periods.
            Keys represent time periods (for example years) and values are decision
            matrices. All matrices must have the same shape, where rows represent
            alternatives and columns represent criteria.

        weights : np.ndarray
            Criteria weights.

        types : np.ndarray
            Criteria types. Use 1 for profit criteria and -1 for cost criteria.

        variability_function : Callable[[np.ndarray], np.ndarray], optional
            Function used to calculate variability values for alternatives based on
            preference values obtained in consecutive periods. The function should
            accept a matrix with periods in rows and alternatives in columns and return
            a one-dimensional array of variability values.

            If ``None``, standard deviation from the DARIA method is used.

            Examples include:

            - ``DARIA()._std``
            - ``DARIA()._gini``
            - ``DARIA()._entropy``
            - ``DARIA()._stat_var``
            - ``DARIA()._coeff_var``

        preference_functions : list[Callable], optional
            Preference functions used by the PROMETHEE II method for each criterion.
            If None, default PROMETHEE preference function (usual) is applied.

        ps : list[float | int], optional
            Preference thresholds used by selected PROMETHEE preference functions.
            If None, default PROMETHEE preference thresholds are applied.

        qs : list[float | int], optional
            Indifference thresholds used by selected PROMETHEE preference functions.
            If None, default PROMETHEE indifference thresholds are applied.

        alt_names : list[str], optional
            Names of alternatives. If provided, they are assigned as row labels in the
            returned DataFrame containing PROMETHEE II net flows.

        Returns
        -------
        tuple[np.ndarray, tuple[np.ndarray, np.ndarray, pd.DataFrame]]
            updated_scores : np.ndarray 
                Final temporal preference scores obtained after updating 
                the most recent PROMETHEE II net flows using DARIA variability 
                information.
            tuple[np.ndarray, np.ndarray, pd.DataFrame]
                variability : np.ndarray 
                    Variability values for alternatives calculated with the selected 
                    variability function.
                direction : np.ndarray 
                    Numerical direction of preference changes over time.
                    Values equal to 1 indicate improvement, -1 indicate deterioration,
                    and 0 indicate stability.
                all_net_flows_df : pd.DataFrame 
                    DataFrame containing PROMETHEE II net flows for all analyzed 
                    periods. Rows correspond to alternatives and columns 
                    correspond to time periods.

        Raises
        ------
        ValueError
            If no decision matrices are provided.

        ValueError
            If decision matrices do not have identical shapes.

        ValueError
            If the number of alternative names does not match the number of
            alternatives.

        TypeError
            If variability_function is not callable.

        Examples
        --------
        >>> import numpy as np
        >>> from pyrepo_mcda.mcda_methods import Temporal_PROMETHEE_II, DARIA

        >>> matrices = {
        ...     "2022": np.array([
        ...         [10, 5],
        ...         [12, 7],
        ...         [8, 9]
        ...     ]),
        ...     "2023": np.array([
        ...         [11, 6],
        ...         [13, 7],
        ...         [9, 8]
        ...     ]),
        ...     "2024": np.array([
        ...         [12, 7],
        ...         [14, 8],
        ...         [10, 9]
        ...     ])
        ... }

        >>> weights = np.array([0.6, 0.4])
        >>> types = np.array([1, -1])

        >>> tp = Temporal_PROMETHEE_II()
        >>> daria = DARIA()

        >>> scores, (variability, direction, net_flows) = tp(
        ...     matrices=matrices,
        ...     weights=weights,
        ...     types=types,
        ...     variability_function=daria._gini
        ... )

        >>> scores
        array(...)
        """

        # Ensure that at least one decision matrix is provided
        if not matrices:
            raise ValueError("At least one decision matrix must be provided.")
        
        # Ensure that all decision matrices contain the same number of
        # alternatives and criteria
        shape = next(iter(matrices.values())).shape

        for year, matrix in matrices.items():
            if matrix.shape != shape:
                raise ValueError(
                    "All decision matrices must have the same shape."
                )

        # Create the PROMETHEE II method object
        promethee_II = PROMETHEE_II()

        # Calculate PROMETHEE II net flows for all analyzed periods
        all_net_flows = {}

        for year, matrix in matrices.items():
            net_flows = promethee_II(matrix, weights, types, preference_functions, ps, qs)
            all_net_flows[year] = net_flows

        all_net_flows_df = pd.DataFrame(all_net_flows)

        # Assign user-defined alternative names and verify that their number
        # matches the number of alternatives
        if alt_names is not None:
            if len(alt_names) != len(all_net_flows_df.index):
                raise ValueError(
                    "The number of alternative names must match the number of alternatives."
                )
            all_net_flows_df.index = alt_names

        all_net_flows_df = all_net_flows_df.rename_axis("Alternatives")


        # perform DARIA calculations

        # years are placed in rows, and alternatives in columns
        matrix = all_net_flows_df.T.to_numpy()

        # PROMETHEE II uses descending preference ordering
        preference_type = 1

        # Calculate efficiencies variability using chosen variability methods from DARIA class
        # Create the DARIA class object
        daria = DARIA()
        # Calculate variability values for each alternative with chosen variability method from DARIA class
        if variability_function is None:
            variability_function = daria._std

        if not callable(variability_function):
            raise TypeError(
                "variability_function must be a callable accepting a matrix."
            )

        variability = variability_function(matrix)
        # Calculate variability directions for each alternative using the method from DARIA class
        _, direction = daria._direction(matrix, preference_type)

        # The most recent year will be updated by variability
        latest_net_flows = all_net_flows_df[
            all_net_flows_df.columns[-1]
        ].to_numpy()

        # update efficiencies using the method from DARIA class
        updated_scores = daria._update_efficiency(
            latest_net_flows,
            variability,
            direction
        )

        return updated_scores, (variability, direction, all_net_flows_df)
