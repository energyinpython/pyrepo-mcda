import unittest
import numpy as np

from pyrepo_mcda.mcda_methods import CRADIS, AHP, MARCOS, PROMETHEE_II, PROSA_C, SAW, ARAS, COPRAS, COCOSO, VMCM, PVM

from pyrepo_mcda import normalizations as norms
from pyrepo_mcda.additions import rank_preferences


# Test for the PROMETHEE II method
class Test_PROMETHEE_II(unittest.TestCase):

    def test_promethee_II(self):
        """Test based on paper Papathanasiou, J., & Ploskas, N. (2018). Promethee. In Multiple Criteria Decision 
        Aid (pp. 57-89). Springer, Cham.
        DOI: https://doi.org/10.1007/978-3-319-91648-4_3"""

        matrix = np.array([[8, 7, 2, 1],
        [5, 3, 7, 5],
        [7, 5, 6, 4],
        [9, 9, 7, 3],
        [11, 10, 3, 7],
        [6, 9, 5, 4]])

        weights = np.array([0.4, 0.3, 0.1, 0.2])

        types = np.array([1, 1, 1, 1])

        promethee_II = PROMETHEE_II()
        preference_functions = [promethee_II._linear_function for pf in range(len(weights))]

        p = 2 * np.ones(len(weights))
        q = 1 * np.ones(len(weights))

        test_result = promethee_II(matrix, weights, types, preference_functions = preference_functions, p = p, q = q)
        real_result = np.array([-0.26, -0.52, -0.22, 0.36, 0.7, -0.06])
        self.assertEqual(list(np.round(test_result, 2)), list(real_result))


# Test for the PROMETHEE II method no 2
class Test_PROMETHEE_II_2(unittest.TestCase):

    def test_promethee_II(self):
        """Ziemba, P. (2019). Towards strong sustainability management—A generalized PROSA method. 
        Sustainability, 11(6), 1555.
        DOI: https://doi.org/10.3390/su11061555"""

        matrix = np.array([[38723, 34913, 25596, 34842, 22570, 39773, 19500, 34525, 16486],
        [33207, 32085, 2123, 32095, 1445, 17485, 868, 16958, 958],
        [0, 0.2, 5, 0.2, 0.2, 0.2, 99, 99, 99],
        [3375, 3127, 3547, 3115, 3090, 4135, 3160, 4295, 3653],
        [11.36, 12.78, 12.78, 12.86, 12.86, 17, 12.86, 17, 12.86],
        [-320.9, -148.4, -148.4, -9.9, -9.9, 0, -9.9, 0, -9.9],
        [203.7, 463, 356.2, 552.5, 295, 383, 264, 352, 264],
        [0, 11.7, 44.8, 11.7, 95.9, 95.9, 116.8, 116.8, 164.9],
        [0, 4.9, 10.7, 5.4, 11.2, 11.2, 11.2, 11.2, 11.2],
        [1, 1, 1, 3.5, 4, 4, 4, 4, 4],
        [21.5, 47.9, 27.7, 39.7, 1.5, 22.7, 2.7, 23.9, 1],
        [0, 3.7, 4.5, 10.3, 11.5, 11.5, 11.3, 11.3, 11.4]])

        matrix = matrix.T

        weights = np.array([0.3333, 0.1667, 0.1667, 0.3333, 0.25, 0.75, 1, 1, 0.4, 0.20, 0.40, 1])

        types = np.array([-1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1])

        p = np.array([2100, 5000, 50, 200, 5, 20, 100, 80, 4, 2, 23, 3])

        q = np.array([420, 1000, 10, 40, 1, 7, 50, 20, 1, 1, 4.6, 1])

        weights = weights / np.sum(weights)

        promethee_II = PROMETHEE_II()
        preference_functions = [promethee_II._linear_function for pf in range(len(weights))]

        test_result = promethee_II(matrix, weights, types, preference_functions = preference_functions, p = p, q = q)
        
        real_result = np.array([-0.4143, -0.5056, -0.2317, -0.2892, 0.3313, 0.0619, 0.4296, 0.1626, 0.4554])

        self.assertEqual(list(np.round(test_result, 4)), list(real_result))


# Test for the PROMETHEE II method no 3
class Test_PROMETHEE_II_3(unittest.TestCase):

    def test_promethee_II(self):
        """Ziemba, P., Wątróbski, J., Zioło, M., & Karczmarczyk, A. (2017). Using the PROSA method 
        in offshore wind farm location problems. Energies, 10(11), 1755.
        DOI: https://doi.org/10.3390/en10111755"""

        matrix = np.array([[16347, 14219, 8160, 8160],
        [9, 8.5, 9, 8.5],
        [73.8, 55, 64.8, 62.5],
        [36.7, 36, 28.5, 29.5],
        [1.5, 2, 2, 1.5],
        [3730, 3240, 1860, 1860],
        [2, 1, 2, 1],
        [1, 1, 2, 3],
        [38.8, 33.1, 45.8, 27.3],
        [4, 2, 4, 3],
        [1720524, 1496512, 858830, 858830],
        [40012, 34803, 19973, 19973]])

        matrix = matrix.T

        weights = np.array([20, 5, 5, 1.67, 1.67, 11.67, 11.67, 5, 5, 16.67, 8.33, 8.33])
        weights = weights / np.sum(weights)

        types = np.array([-1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1])

        p = np.array([7280, 4, 13.4, 7.4, 3, 1662, 3, 3, 13.8, 3, 766240, 17820])

        promethee_II = PROMETHEE_II()
        preference_functions = [promethee_II._vshape_function for pf in range(len(weights))]

        test_result = promethee_II(matrix, weights, types, preference_functions = preference_functions, p = p)
        
        real_result = np.array([-0.0445, 0.1884, -0.0954, -0.0485])

        self.assertEqual(list(np.round(test_result, 4)), list(real_result))


# Test for the PROSA-C method (PROSA Examining Sustainability at the Criteria Level)
class Test_PROSA_C(unittest.TestCase):

    def test_prosa_c(self):
        """Ziemba, P. (2019). Towards strong sustainability management—A generalized PROSA method. 
        Sustainability, 11(6), 1555.
        DOI: https://doi.org/10.3390/su11061555"""

        matrix = np.array([[38723, 34913, 25596, 34842, 22570, 39773, 19500, 34525, 16486],
        [33207, 32085, 2123, 32095, 1445, 17485, 868, 16958, 958],
        [0, 0.2, 5, 0.2, 0.2, 0.2, 99, 99, 99],
        [3375, 3127, 3547, 3115, 3090, 4135, 3160, 4295, 3653],
        [11.36, 12.78, 12.78, 12.86, 12.86, 17, 12.86, 17, 12.86],
        [-320.9, -148.4, -148.4, -9.9, -9.9, 0, -9.9, 0, -9.9],
        [203.7, 463, 356.2, 552.5, 295, 383, 264, 352, 264],
        [0, 11.7, 44.8, 11.7, 95.9, 95.9, 116.8, 116.8, 164.9],
        [0, 4.9, 10.7, 5.4, 11.2, 11.2, 11.2, 11.2, 11.2],
        [1, 1, 1, 3.5, 4, 4, 4, 4, 4],
        [21.5, 47.9, 27.7, 39.7, 1.5, 22.7, 2.7, 23.9, 1],
        [0, 3.7, 4.5, 10.3, 11.5, 11.5, 11.3, 11.3, 11.4]])

        matrix = matrix.T

        weights = np.array([0.3333, 0.1667, 0.1667, 0.3333, 0.25, 0.75, 1, 1, 0.4, 0.20, 0.40, 1])

        types = np.array([-1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1])

        p = np.array([2100, 5000, 50, 200, 5, 20, 100, 80, 4, 2, 23, 3])

        q = np.array([420, 1000, 10, 40, 1, 7, 50, 20, 1, 1, 4.6, 1])

        weights = weights / np.sum(weights)

        s = np.array([0.4, 0.5, 0.3, 0.4, 0.3, 0.4, 0.3, 0.3, 0.2, 0.4, 0.4, 0.2])

        prosa_c = PROSA_C()
        preference_functions = [prosa_c._linear_function for pf in range(len(weights))]

        test_result = prosa_c(matrix, weights, types, preference_functions = preference_functions, p = p, q = q, s = s)
        
        real_result = np.array([-0.5921, -0.6014, -0.3240, -0.4381, 0.2791, -0.0703, 0.3739, 0.0451, 0.3592])

        self.assertEqual(list(np.round(test_result, 4)), list(real_result))



# Test for the AHP method
class Test_AHP(unittest.TestCase):

    def test_ahp(self):
        """Papathanasiou, J., & Ploskas, N. (2018). Ahp. In Multiple Criteria Decision Aid 
        (pp. 109-129). Springer, Cham.
        DOI: https://doi.org/10.1007/978-3-319-91648-4_5"""

        # input data
        # user gives matrices of comparison for criteria
        # step 1 pairwise comparison matrix of the criteria
        PCcriteria = np.array([[1, 1, 5, 3], [1, 1, 5, 3], 
        [1/5, 1/5, 1, 1/3], [1/3, 1/3, 3, 1]])

        # and for alternatives
        # step 4 pairwise comparison matrix of the alternatives
        PCM1 = np.array([[1, 5, 1, 1, 1/3, 3],
        [1/5, 1, 1/3, 1/5, 1/7, 1],
        [1, 3, 1, 1/3, 1/5, 1],
        [1, 5, 3, 1, 1/3, 3],
        [3, 7, 5, 3, 1, 7],
        [1/3, 1, 1, 1/3, 1/7, 1]])
        PCM2 = np.array([[1, 7, 3, 1/3, 1/3, 1/3],
        [1/7, 1, 1/3, 1/7, 1/9, 1/7],
        [1/3, 3, 1, 1/5, 1/5, 1/5],
        [3, 7, 5, 1, 1, 1],
        [3, 9, 5, 1, 1, 1],
        [3, 7, 5, 1, 1, 1]])
        PCM3 = np.array([[1, 1/9, 1/7, 1/9, 1, 1/5],
        [9, 1, 1, 1, 5, 3],
        [7, 1, 1, 1, 5, 1],
        [9, 1, 1, 1, 7, 3],
        [1, 1/5, 1/5, 1/7, 1, 1/3],
        [5, 1/3, 1, 1/3, 3, 1]])
        PCM4 = np.array([[1, 1/5, 1/5, 1/3, 1/7, 1/5],
        [5, 1, 1, 3, 1/3, 1],
        [5, 1, 1, 1, 1/3, 1],
        [3, 1/3, 1, 1, 1/7, 1],
        [7, 3, 3, 7, 1, 5],
        [5, 1, 1, 1, 1/5, 1]])

        ahp = AHP()
        # step 2 check consistency of matrix with criteria comparison
        ahp._check_consistency(PCcriteria)
        
        # step 3 compute priority vector of criteria (weights)
        weights = ahp._calculate_eigenvector(PCcriteria)
        
        # step 4 form pairwise comparison matrices of the alternatives for each criterion
        alt_matrices = []
        alt_matrices.append(PCM1)
        alt_matrices.append(PCM2)
        alt_matrices.append(PCM3)
        alt_matrices.append(PCM4)

        # step 5 consistency check of pairwise comparison matrices of the alternatives

        # step 6 compute local priority vectors of alternatives
        # select the method to calculate priority vector
        # the default method to calculate priority vector is ahp._calculate_eigenvector
        # calculate_priority_vector_method = ahp._calculate_eigenvector
        test_result = ahp._classic_ahp(alt_matrices, weights)
        
        real_result = np.array([0.117, 0.071, 0.095, 0.212, 0.350, 0.155])

        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for the SAW method
class Test_SAW(unittest.TestCase):

    def test_saw(self):
        """Rizan, O., Wahyuningsih, D., Pradana, H. A., & Ramadella, S. (2020, October). 
        SAW Method in Supporting the Process of Admission of New Junior High School Students. 
        In 2020 8th International Conference on Cyber and IT Service Management (CITSM) (pp. 1-5). 
        IEEE.
        DOI: https://doi.org/10.1109/CITSM50537.2020.9268874"""

        matrix = np.array([[0.75, 0.50, 0.75, 0, 0, 0, 1],
        [0.75, 1, 0.75, 0, 0, 0, 0.75],
        [0.75, 0.75, 0.75, 0, 0.50, 0.25, 1],
        [0.50, 0.50, 0.75, 1, 0.50, 0, 0.75]])

        weights = np.array([0.1, 0.1, 0.1, 0.15, 0.2, 0.25, 0.1])

        types = np.array([1, 1, 1, 1, 1, 1, 1])

        saw = SAW(normalization_method=norms.linear_normalization)
        pref = saw(matrix, weights, types)

        test_result = saw(matrix, weights, types)
        
        real_result = np.array([0.35, 0.375, 0.825, 0.642])
        
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for the AHP method no 2
# usage with matrix including numerical values of performances
# and vector with weights given in numerical values
class Test_AHP_2(unittest.TestCase):

    def test_ahp(self):
        """Rizan, O., Wahyuningsih, D., Pradana, H. A., & Ramadella, S. (2020, October). 
        SAW Method in Supporting the Process of Admission of New Junior High School Students. 
        In 2020 8th International Conference on Cyber and IT Service Management (CITSM) (pp. 1-5). 
        IEEE.
        DOI: https://doi.org/10.1109/CITSM50537.2020.9268874
        Test for AHP when the matrix with numerical performance values
        and vector with numerical criteria weights are provided and AHP
        is used like the SAW method"""

        matrix = np.array([[0.75, 0.50, 0.75, 0, 0, 0, 1],
        [0.75, 1, 0.75, 0, 0, 0, 0.75],
        [0.75, 0.75, 0.75, 0, 0.50, 0.25, 1],
        [0.50, 0.50, 0.75, 1, 0.50, 0, 0.75]])

        weights = np.array([0.1, 0.1, 0.1, 0.15, 0.2, 0.25, 0.1])

        types = np.array([1, 1, 1, 1, 1, 1, 1])

        ahp = AHP(normalization_method=norms.linear_normalization)

        test_result = ahp(matrix, weights, types)
        
        real_result = np.array([0.35, 0.375, 0.825, 0.642])
        
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for the MARCOS method
class Test_MARCOS(unittest.TestCase):

    def test_marcos(self):
        """Test based on paper Stević, Ž., Pamučar, D., Puška, A., & Chatterjee, P. (2020). 
        Sustainable supplier selection in healthcare industries using a new MCDM method: 
        Measurement of alternatives and ranking according to COmpromise solution (MARCOS). 
        Computers & Industrial Engineering, 140, 106231.
        DOI: https://doi.org/10.1016/j.cie.2019.106231"""

        matrix = np.array([[6.257, 4.217, 4.217, 6.257, 3.000, 4.217, 5.000, 3.557, 3.557, 3.557, 3.000, 5.000, 4.718, 3.557, 3.557, 2.080, 3.557, 3.000, 4.718, 3.557, 2.080],
        [4.217, 6.804, 7.000, 5.000, 7.000, 6.804, 5.593, 5.593, 6.804, 7.000, 5.000, 7.612, 5.593, 6.257, 5.000, 7.000, 5.593, 5.593, 6.804, 5.593, 5.593],
        [4.718, 5.593, 6.257, 5.000, 4.718, 5.000, 5.593, 4.718, 5.593, 5.000, 3.557, 6.257, 5.000, 4.718, 4.718, 5.000, 3.557, 5.593, 5.593, 3.557, 4.217],
        [5.000, 6.804, 5.000, 3.000, 5.000, 6.257, 7.612, 3.557, 5.000, 6.257, 6.257, 5.593, 6.257, 5.000, 5.593, 7.000, 5.000, 6.257, 5.000, 3.557, 4.217],
        [3.557, 5.593, 6.804, 3.000, 5.000, 7.000, 5.593, 5.000, 6.257, 7.000, 5.593, 7.612, 6.257, 6.257, 5.000, 6.257, 5.593, 7.000, 5.000, 4.718, 5.000],
        [6.257, 3.000, 4.217, 5.000, 3.557, 3.000, 4.217, 3.000, 4.217, 3.000, 2.466, 3.000, 4.217, 3.557, 5.000, 3.000, 4.217, 3.557, 2.080, 5.000, 3.000],
        [4.217, 5.000, 6.257, 5.593, 3.557, 5.593, 4.217, 5.593, 5.000, 6.257, 3.557, 5.000, 6.257, 5.593, 5.593, 7.000, 6.257, 5.000, 6.257, 5.593, 5.000],
        [7.612, 1.442, 3.000, 9.000, 2.080, 3.000, 1.442, 2.080, 1.442, 3.000, 1.000, 1.442, 2.080, 3.000, 3.000, 1.000, 1.442, 1.442, 3.000, 2.080, 1.000]])

        weights = np.array([0.127, 0.159, 0.060, 0.075, 0.043, 0.051, 0.075, 0.061, 0.053, 0.020, 0.039, 0.022, 0.017, 0.027, 0.022, 0.039, 0.017, 0.035, 0.015, 0.024, 0.016])

        types = np.array([-1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        method = MARCOS()
        test_result = method(matrix, weights, types)
        real_result = np.array([0.524, 0.846, 0.703, 0.796, 0.843, 0.499, 0.722, 0.291])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for the CRADIS method
class Test_CRADIS(unittest.TestCase):

    def test_cradis(self):
        """Test based on Puška, A., Stević, Ž., & Pamučar, D. (2021). Evaluation and selection 
        of healthcare waste incinerators using extended sustainability criteria and multi-criteria 
        analysis methods. Environment, Development and Sustainability, 1-31.
        DOI: https://doi.org/10.1007/s10668-021-01902-2
        """

        matrix = np.array([[1.82, 1.59, 2.62, 2.62, 4.31, 3.3, 2.29, 3.3, 4.31, 5.31, 2.29, 1.26, 0.36, 30, 10, 5.02],
        [1.82, 1.59, 2.62, 2.62, 3.63, 3.3, 2.29, 3.3, 4.31, 6., 2.29, 1.26, 0.54, 40., 11.5, 6.26],
        [2.88, 2.62, 3.3, 3., 4.64, 3.91, 2.52, 3.91, 3.3, 6., 3.3, 1.44, 0.75, 50., 12.5, 8.97],
        [1.82, 1.59, 2.62, 3.17, 3.63, 3.3, 2.29, 3.3, 4.31, 6., 3.3, 2., 0.57, 65., 17.5, 8.79],
        [3.11, 3., 3.91, 4., 5., 4.58, 3.3, 4., 2.29, 5., 3.91, 2.88, 1.35, 100., 16.5, 11.68],
        [2.88, 2.29, 3.63, 3.63, 5., 4.31, 3.3, 4.31, 2.88, 6., 4.31, 2.29, 1.2, 100., 15.5, 12.9]])

        weights = np.array([0.07, 0.05, 0.05, 0.06, 0.09, 0.06, 0.06, 0.06, 0.05, 0.07, 0.05, 0.05, 0.09, 0.06, 0.07, 0.06])

        types = np.array([-1., -1., -1., -1., -1., -1., -1., -1.,  1.,  1., -1., -1.,  1.,  1., -1., -1.])

        method = CRADIS()
        test_result = method(matrix, weights, types)
        
        real_result = np.array([0.7932, 0.8187, 0.6393, 0.7380, 0.5807, 0.6123])
        self.assertEqual(list(np.round(test_result, 2)), list(np.round(real_result, 2)))


# Test for the ARAS method
class Test_ARAS(unittest.TestCase):

    def test_aras(self):
        """Test based on paper Goswami, S., & Mitra, S. (2020). Selecting the best mobile model by applying 
        AHP-COPRAS and AHP-ARAS decision making methodology. International Journal of Data and 
        Network Science, 4(1), 27-42.
        DOI: http://dx.doi.org/10.5267/j.ijdns.2019.8.004"""

        matrix = np.array([[80, 16, 2, 5],
        [110, 32, 2, 9],
        [130, 64, 4, 9],
        [185, 64, 4, 1],
        [135, 64, 3, 4],
        [140, 32, 3, 5],
        [185, 64, 6, 7],
        [110, 16, 3, 3],
        [120, 16, 4, 3],
        [340, 128, 6, 5]])

        weights = np.array([0.60338, 0.13639, 0.19567, 0.06456])

        types = np.array([-1, 1, 1, 1])

        method = ARAS()
        test_result = method(matrix, weights, types)
        real_result = np.array([0.68915, 0.58525, 0.62793, 0.46666, 0.54924, 0.49801, 0.56959, 0.54950, 0.54505, 0.53549])
        self.assertEqual(list(np.round(test_result, 5)), list(real_result))


# Test for the COPRAS method
class Test_COPRAS(unittest.TestCase):

    def test_copras(self):
        """Test based on paper Goswami, S., & Mitra, S. (2020). Selecting the best mobile model by applying 
        AHP-COPRAS and AHP-ARAS decision making methodology. International Journal of Data and 
        Network Science, 4(1), 27-42.
        DOI: http://dx.doi.org/10.5267/j.ijdns.2019.8.004"""

        matrix = np.array([[80, 16, 2, 5],
        [110, 32, 2, 9],
        [130, 64, 4, 9],
        [185, 64, 4, 1],
        [135, 64, 3, 4],
        [140, 32, 3, 5],
        [185, 64, 6, 7],
        [110, 16, 3, 3],
        [120, 16, 4, 3],
        [340, 128, 6, 5]])

        weights = np.array([0.60338, 0.13639, 0.19567, 0.06456])

        types = np.array([-1, 1, 1, 1])

        method = COPRAS()
        test_result = method(matrix, weights, types)
        real_result = np.array([1, 0.85262, 0.91930, 0.68523, 0.80515, 0.72587, 0.83436, 0.79758, 0.79097, 0.79533])
        self.assertEqual(list(np.round(test_result, 5)), list(real_result))


# Test for the COCOSO method
class Test_COCOSO(unittest.TestCase):

    def test_cocoso(self):
        """Test based on paper Yazdani, M., Zarate, P., Kazimieras Zavadskas, E., & Turskis, Z. (2019). 
        A combined compromise solution (CoCoSo) method for multi-criteria decision-making problems. 
        Management Decision, 57(9), 2501-2519.
        DOI: https://doi.org/10.1108/MD-05-2017-0458"""

        matrix = np.array([[60, 0.4, 2540, 500, 990],
        [6.35, 0.15, 1016, 3000, 1041],
        [6.8, 0.1, 1727.2, 1500, 1676],
        [10, 0.2, 1000, 2000, 965],
        [2.5, 0.1, 560, 500, 915],
        [4.5, 0.08, 1016, 350, 508],
        [3, 0.1, 1778, 1000, 920]])

        weights = np.array([0.036, 0.192, 0.326, 0.326, 0.12])

        types = np.array([1, -1, 1, 1, 1])

        method = COCOSO()
        test_result = method(matrix, weights, types)
        real_result = np.array([2.041, 2.788, 2.882, 2.416, 1.3, 1.443, 2.52])
        self.assertEqual(list(np.round(test_result, 1)), list(np.round(real_result, 1)))


# Test for the PVM method
class Test_PVM(unittest.TestCase):

    def test_pvm(self):
        """Nermend, K. (2023). The Issue of Multi-criteria and Multi-dimensionality in Decision Support. 
        In: Multi-Criteria and Multi-Dimensional Analysis in Decisions . Vector Optimization. Springer, 
        Cham. https://doi.org/10.1007/978-3-031-40538-9_1"""

        matrix = np.array([
            [326, 27.5, 24.3, 48.83],
            [253, 22, 18.3, 51.11],
            [561, 26, 50, 52.42],
            [405, 21.7, 35.6, 29.12],
            [479, 18, 41, 64.08]
        ])

        weights = np.array([0.1667, 0.1667, 0.3333, 0.3333])

        types = ['dm', 'd', 'd', 'dm']

        psi = np.array([300, 21.5, 18.3, 41.5325])

        phi = np.array([500, 18, 20, 52.0925])

        method = PVM()
        test_result = method(matrix, weights, types, psi = psi, phi = phi)
        
        real_result = np.array([-0.0004,  0.019,  -0.0715, -0.0156, -0.0529])
        self.assertEqual(list(np.round(test_result, 4)), list(np.round(real_result, 4)))


def main():
    test_promethee_II = Test_PROMETHEE_II()
    test_promethee_II.test_promethee_II()

    test_promethee_II = Test_PROMETHEE_II_2()
    test_promethee_II.test_promethee_II()

    test_promethee_II = Test_PROMETHEE_II_3()
    test_promethee_II.test_promethee_II()

    test_prosa_c = Test_PROSA_C()
    test_prosa_c.test_prosa_c()

    test_ahp = Test_AHP()
    test_ahp.test_ahp()

    test_saw = Test_SAW()
    test_saw.test_saw()

    test_ahp_2 = Test_AHP_2()
    test_ahp_2.test_ahp()

    test_marcos = Test_MARCOS()
    test_marcos.test_marcos()

    test_cradis = Test_CRADIS()
    test_cradis.test_cradis()

    test_aras = Test_ARAS()
    test_aras.test_aras()

    test_copras = Test_COPRAS()
    test_copras.test_copras()

    test_cocoso = Test_COCOSO()
    test_cocoso.test_cocoso()

    test_pvm = Test_PVM()
    test_pvm.test_pvm()


if __name__ == '__main__':
    main()
