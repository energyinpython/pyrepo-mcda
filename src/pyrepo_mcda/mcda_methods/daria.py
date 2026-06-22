import copy
import itertools

import numpy as np

from pyrepo_mcda import normalizations as norms


class DARIA():
    """
    Data vARIability Assessment (DARIA) method for analyzing variability 
    and temporal changes in alternative preferences.
    """
    
    def __init__(self):
        """
        Create the DARIA object
        """

        pass


    # Gini coefficient variability measure
    def _gini(self, R):
        """
        Calculate variability values measured by the Gini coefficient in scores obtained by each evaluated option.
        
        Parameters
        -----------
            R : ndarray
                Matrix with preference values obtained with MCDA method (for example, TOPSIS)
                with `t` periods of time in rows and `m` alternatives in columns.

        Returns
        --------
            ndarray
                Vector with Gini coefficient values for each alternative.

        Examples
        ----------
        >>> daria = DARIA()
        >>> variability = daria._gini(matrix)
        """

        # t is number of periods of time which are in rows
        # m is number of alternatives which are in columns
        t, m = np.shape(R)
        # G is vector for Gini coeff values for each alternative
        G = np.zeros(m)
        # iteration over alternatives i=1, 2, ..., m
        # iteration over periods p=1, 2, ..., t
        # Calculate `G` Gini coefficient value of preferences in all periods for each alternative
        for i in range(m):
            Yi = 0
            # iteration over alternatives i = 1, 2, ..., m
            if np.mean(R[:, i]) != 0:
                for p in range(t):
                    Yi += np.sum(np.abs(R[p, i] - R[:, i]) / (2 * t**2 * (np.sum(R[:, i]) / t)))
            else:
                for p in range(t):
                    Yi += np.sum(np.abs(R[p, i] - R[:, i]) / (t**2 - t))

            G[i] = Yi
        return G

    
    # Entropy variability measure
    def _entropy(self, R):
        """
        
        Calculate variability values measured by the Entropy in scores obtained by each evaluated option.
        
        Parameters
        -----------
            R : ndarray
                Matrix with preference values obtained with MCDA method (for example, TOPSIS)
                with `t` periods of time in rows and `m` alternatives in columns.
        
        Returns
        --------
            ndarray
                Vector with Entropy values for each alternative.
        
        Examples
        ----------
        >>> daria = DARIA()
        >>> variability = daria._entropy(matrix)
        """

        # normalize the decision matrix with sum_normalization method from normalizations as for profit criteria
        criteria_type = np.ones(np.shape(R)[1])
        pij = norms.sum_normalization(R, criteria_type)
        # Transform negative values in decision matrix X to positive values
        pij = np.abs(pij)
        m, n = np.shape(pij)
        H = np.zeros((m, n))

        # Calculate entropy
        for j, i in itertools.product(range(n), range(m)):
            if pij[i, j]:
                H[i, j] = pij[i, j] * np.log(pij[i, j])

        h = np.sum(H, axis = 0) * (-1 * ((np.log(m)) ** (-1)))
        d = 1 - h
        return d


    # standard deviation variability measure
    def _std(self, R):
        """
        Calculate variability values measured by the Standard Deviation in scores obtained by each evaluated option.
        
        Parameters
        -----------
        R : ndarray
            Matrix with preference values obtained with MCDA method (for example, TOPSIS)
            with `t` periods of time in rows and `m` alternatives in columns.
        
        Returns
        --------
        ndarray
            Vector with Standard Deviation values for each alternative.
        
        Examples
        ----------
        >>> daria = DARIA()
        >>> variability = daria._std(matrix)
        """

        # Calculate the standard deviation of each criterion in decision matrix
        stdv = np.sqrt((np.sum(np.square(R - np.mean(R, axis = 0)), axis = 0)) / R.shape[0])
        # std = np.std(R, axis=0, ddof=1)
        return stdv


    # statistical variance variability measure
    def _stat_var(self, X):
        """
        Calculate variability values measured by the Statistical Variance in scores obtained by each evaluated option.
        
        Parameters
        ----------
        R : ndarray
            Matrix with preference values obtained with MCDA method (for example, TOPSIS)
            with `t` periods of time in rows and `m` alternatives in columns.
        
        Returns
        -------
        ndarray
            Vector with Statistical Variance values for each alternative.
        
        Examples
        --------
        >>> daria = DARIA()
        >>> variability = daria._stat_var(matrix)
        """
        
        # Calculate the statistical variance for each criterion
        v = np.mean(np.square(X - np.mean(X, axis = 0)), axis = 0)
        return v


    # Coefficient of variation
    def _coeff_var(self, X):
        """
        Calculate variability values measured by the Coefficient of Variation in scores obtained by each evaluated option.
        
        Parameters
        ----------
        R : ndarray
            Matrix with preference values obtained with MCDA method (for example, TOPSIS)
            with `t` periods of time in rows and `m` alternatives in columns.
        
        Returns
        -------
        ndarray
            Vector with Coefficient of Variation values for each alternative.
        
        Examples
        --------
        >>> daria = DARIA()
        >>> variability = daria._coeff_var(matrix)
        """

        m, n = X.shape
        # Normalize the decision matrix X with sum_normalization method from normalizations
        criteria_types = np.ones(n)
        B = norms.sum_normalization(X, criteria_types)

        # Calculate the standard deviation of each column
        Bm = np.sum(B, axis = 0) / m
        std = np.sqrt(np.sum(((B - Bm)**2), axis = 0) / (m - 1))

        # Calculate the Coefficient of Variation for each criterion
        ej = std / Bm
        return ej


    # Direction of variability
    # for MCDA methods preference_type = 1: descending order: higher is better, preference_type -1: opposite
    def _direction(self, R, preference_type = 1):
        """
        Determine the direction of the variability of alternatives scores obtained in the following 
        periods of time.
        
        Parameters
        ----------
        R : ndarray
            Matrix with preference values obtained with MCDA method (for example, TOPSIS)
            with `t` periods of time in rows and `m` alternatives in columns.
        preference_type : int
            The variable represents the ordering of alternatives by the MCDA method. It can be equal to
            1 or -1. 1 means that the MCDA method sorts options in descending order
            according to preference values (for example, the TOPSIS method). -1 means that 
            the MCDA method sorts options in ascending order according to preference values 
            (for example, the VIKOR method).
        
        Returns
        -------
        direction_list : list
            List with strings representing the direction of variability in the form of the
            arrow up for improvement, arrow down for worsening, and = for stability.
            It is useful for results presentation.
        dir_class : ndarray
            Vector with numerical values representing the direction of variability. 1 represents
            increasing preference values, and -1 means decreasing preference values.
            It is used to calculate final aggregated preference values using DARIA method in
            next stage of DARIA method.
        
        Examples
        --------
        >>> daria = DARIA()
        >>> dir_list, dir_class = daria._direction(matrix, preference_type)
        """

        t, m = np.shape(R)
        direction_list = []
        dir_class = np.zeros(m)
        # iteration over alternatives i = 1, 2, ..., m
        for i in range(m):
            thresh = 0
            # iteration over periods p = 1, 2, ..., t
            for p in range(1, t):
                thresh += R[p, i] - R[p - 1, i]
            # classification based on thresh
            dir_class[i] = np.sign(thresh)
            
        direction_array = copy.deepcopy(dir_class)
        direction_array = direction_array * preference_type
        for i in range(len(direction_array)):
            if direction_array[i] == 1:
                direction_list.append(r'$\uparrow$')
            elif direction_array[i] == -1:
                direction_list.append(r'$\downarrow$')
            elif direction_array[i] == 0:
                direction_list.append(r'$=$')
        return direction_list, dir_class


    def _update_efficiency(self, scores, variability, direction):
        """
        Calculate final aggregated preference values of alternatives of DARIA method.
        Obtained preference values can be sorted according to chosen MCDA method rule to generate
        ranking of alternatives.
        
        Parameters
        ----------
        scores : ndarray
            Vector with preference values of alternatives from the most recent year analyzed
            obtained by chosen MCDA method.
        variability : ndarray
            Vector with variability values of alternatives preferences obtained in investigated
            periods.
        direction : ndarray
            Vector with numerical values of the direction of variability in values of alternatives 
            preferences obtained in investigated periods. 1 represents increasing in following
            preference values, and -1 means decreasing in following preference values.
        
        Returns
        -------
        ndarray
            Final aggregated preference values of alternatives considering variability in
            preference values obtained in the following periods.
        
        Examples
        --------
        >>> updated_scores = daria._update_efficiency(scores, variability, direction)
        >>> rank = rank_preferences(updated_scores, reverse = True)
        """

        return scores + variability * direction
