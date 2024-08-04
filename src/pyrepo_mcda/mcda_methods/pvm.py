import numpy as np

class PVM():
    def __init__(self):
        """
        Create the PVM method object
        """
        pass

    def __call__(self, matrix, weights, types, psi = None, phi = None):
        """
        Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.

        Parameters
        -----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Attribute weights to the criteria. Sum of weights must be equal to 1.
            types: list
                Define the criteria character:
                Motivating criteria are represented by `m`
                Demotivating criteria are represented by `dm`
                Desirable criteria are represented by `d`
                Non-desirable criteria are represented by `nd`
            psi: ndarray
                Preference vector with motivating or desirable values. Providing of psi is optional.
            phi: ndarray
                Preference vector with demotivating or non-desirable values. Providing of psi is optional.

        Returns
        --------
            ndrarray
                Preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ----------
        >>> pvm = PVM()
        >>> pref = pvm(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        
        return PVM._pvm(matrix, weights, types, psi, phi)

    @staticmethod
    def _pvm(matrix, weights, types, psi, phi):

        # Determining the preference vector
        if (psi is None) or (phi is None):
            psi = np.zeros(matrix.shape[1])
            phi = np.zeros(matrix.shape[1])

            for i in range(len(types)):
                if types[i] == 'm':
                    psi[i] = np.quantile(matrix[:,i], 0.75)
                    phi[i] = np.quantile(matrix[:,i], 0.25)
                elif types[i] == 'dm':
                    psi[i] = np.quantile(matrix[:,i], 0.25)
                    phi[i] = np.quantile(matrix[:,i], 0.75)
                elif types[i] == 'd':
                    psi[i] = np.max(matrix[:, i])
                    phi[i] = np.min(matrix[:, i])
                elif types[i] == 'nd':
                    psi[i] = np.min(matrix[:, i])
                    phi[i] = np.max(matrix[:, i])
                else:
                    raise ValueError('Only `m`, `dm`, `d` and `nd` criteria types are accepted.')

        # Normalization of decision matrix
        nmatrix = matrix / np.sqrt(np.sum(np.square(matrix), axis = 0))

        # Normalization of preference vector
        psi_prim = psi / np.sqrt(np.sum(np.square(matrix), axis = 0))
        phi_prim = phi / np.sqrt(np.sum(np.square(matrix), axis = 0))

        T_prim = psi_prim - phi_prim
        
        T_prim_prim = T_prim / np.sqrt(np.sum(np.square(T_prim)))

        # Determination of importance factor `mi`
        mi_v = np.zeros(matrix.shape[0])
        mi_d = np.zeros(matrix.shape[0])
        mi_nd = np.zeros(matrix.shape[0])
        l_mi_v, l_mi_d, l_mi_nd = 0, 0, 0
        
        for i in range(len(types)):
            if types[i] == 'm' or types[i] == 'dm':
                mi_v += (nmatrix[:,i] - phi_prim[i]) * T_prim_prim[i] * weights[i]
                l_mi_v += 1
            elif types[i] == 'd':
                mi_d += weights[i]**2 * (nmatrix[:,i] - psi_prim[i])**2
                l_mi_d += 1
            elif types[i] == 'nd':
                mi_nd += weights[i]**2 * (nmatrix[:,i] - psi_prim[i])**2
                l_mi_nd += 1

        mi_d = np.sqrt(mi_d)
        mi_nd = np.sqrt(mi_nd)

        mi = (mi_v * l_mi_v - mi_d * l_mi_d + mi_nd * l_mi_nd) / (l_mi_v + l_mi_d + l_mi_nd)

        return mi