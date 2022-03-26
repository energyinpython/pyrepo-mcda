from pyrepo_mcda.mcda_methods import CODAS
from pyrepo_mcda.mcda_methods import TOPSIS
from pyrepo_mcda.mcda_methods import WASPAS
from pyrepo_mcda.mcda_methods import VIKOR
from pyrepo_mcda.mcda_methods import SPOTIS
from pyrepo_mcda.mcda_methods import EDAS
from pyrepo_mcda.mcda_methods import MABAC
from pyrepo_mcda.mcda_methods import MULTIMOORA

from pyrepo_mcda.mcda_methods.multimoora import MULTIMOORA_RS, MULTIMOORA_RP, MULTIMOORA_FMF

from pyrepo_mcda import distance_metrics as dists
from pyrepo_mcda import correlations as corrs
from pyrepo_mcda import normalizations as norms
from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda import compromise_rankings as compromises
from pyrepo_mcda.additions import rank_preferences

import unittest
import numpy as np

import copy


# Test for the VIKOR method
class Test_VIKOR(unittest.TestCase):

    def test_vikor(self):
        """Test based on paper Papathanasiou, J., & Ploskas, N. (2018). Vikor. In Multiple Criteria Decision Aid 
        (pp. 31-55). Springer, Cham."""

        matrix = np.array([[8, 7, 2, 1],
        [5, 3, 7, 5],
        [7, 5, 6, 4],
        [9, 9, 7, 3],
        [11, 10, 3, 7],
        [6, 9, 5, 4]])

        weights = np.array([0.4, 0.3, 0.1, 0.2])
        types = np.array([1, 1, 1, 1])

        method = VIKOR(v = 0.625)
        test_result = method(matrix, weights, types)
        real_result = np.array([0.640, 1.000, 0.693, 0.271, 0.000, 0.694])

        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for the TOPSIS method
class Test_TOPSIS(unittest.TestCase):

    def test_topsis(self):
        """Ersoy, Y. (2021). Equipment selection for an e-commerce company using entropy-based topsis, 
        edas and codas methods during the COVID-19. LogForum, 17(3)."""

        matrix = np.array([[256, 8, 41, 1.6, 1.77, 7347.16],
        [256, 8, 32, 1.0, 1.8, 6919.99],
        [256, 8, 53, 1.6, 1.9, 8400],
        [256, 8, 41, 1.0, 1.75, 6808.9],
        [512, 8, 35, 1.6, 1.7, 8479.99],
        [256, 4, 35, 1.6, 1.7, 7499.99]])

        weights = np.array([0.405, 0.221, 0.134, 0.199, 0.007, 0.034])
        types = np.array([1, 1, 1, 1, -1, -1])

        method = TOPSIS(normalization_method=norms.vector_normalization, distance_metric=dists.euclidean)
        test_result = method(matrix, weights, types)
        real_result = np.array([0.308, 0.254, 0.327, 0.263, 0.857, 0.192])

        self.assertEqual(list(np.round(test_result, 2)), list(np.round(real_result, 2)))


# Test for the CODAS method
class Test_CODAS(unittest.TestCase):

    def test_codas(self):
        """Badi, I., Abdulshahed, A. M., & Shetwan, A. (2018). A case study of supplier selection 
        for a steelmaking company in Libya by using the Combinative Distance-based ASsessment (CODAS) 
        model. Decision Making: Applications in Management and Engineering, 1(1), 1-12."""

        matrix = np.array([[45, 3600, 45, 0.9],
        [25, 3800, 60, 0.8],
        [23, 3100, 35, 0.9],
        [14, 3400, 50, 0.7],
        [15, 3300, 40, 0.8],
        [28, 3000, 30, 0.6]])

        weights = np.array([0.2857, 0.3036, 0.2321, 0.1786])
        types = np.array([1, -1, 1, 1])

        method = CODAS(normalization_method = norms.linear_normalization, distance_metric = dists.euclidean, tau = 0.02)
        test_result = method(matrix, weights, types)
        real_result = np.array([1.3914, 0.3411, -0.2170, -0.5381, -0.7292, -0.2481])
        
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))

    def test_codas2(self):
        """Badi, I., Ballem, M., & Shetwan, A. (2018). Site selection of desalination plant in 
        Libya by using Combinative Distance-based Assessment (CODAS) Method. International Journal 
        for Quality Research, 12(3), 609."""

        matrix = np.array([[8, 8, 10, 9, 5],
        [8, 9, 9, 9, 8],
        [9, 9, 7, 8, 6],
        [8, 8, 7, 8, 9],
        [9, 8, 7, 7, 4]])

        weights = np.array([0.19, 0.26, 0.24, 0.17, 0.14])
        types = np.array([-1, 1, 1, 1, -1])

        method = CODAS(normalization_method = norms.linear_normalization, distance_metric = dists.euclidean, tau = 0.02)
        test_result = method(matrix, weights, types)
        real_result = np.array([0.4463, 0.1658, -0.2544, -0.4618, 0.1041])
        
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))


# Test for the WASPAS method
class Test_WASPAS(unittest.TestCase):

    def test_waspas(self):
        """Tuş, A., & Aytaç Adalı, E. (2019). The new combination with CRITIC and WASPAS methods 
        for the time and attendance software selection problem. Opsearch, 56(2), 528-538."""

        matrix = np.array([[5000, 3, 3, 4, 3, 2],
        [680, 5, 3, 2, 2, 1],
        [2000, 3, 2, 3, 4, 3],
        [600, 4, 3, 1, 2, 2],
        [800, 2, 4, 3, 3, 4]])

        types = np.array([-1, 1, 1, 1, 1, 1])
        weights = mcda_weights.critic_weighting(matrix)

        method = WASPAS(normalization_method=norms.linear_normalization, lambda_param=0.5)
        test_result = method(matrix, weights, types)
        real_result = np.array([0.5622, 0.6575, 0.6192, 0.6409, 0.722])
        
        self.assertEqual(list(np.round(test_result, 3)), list(np.round(real_result, 3)))


# Test for the SPOTIS method
class Test_SPOTIS(unittest.TestCase):

    def test_spotis(self):
        """Test based on paper Dezert, J., Tchamova, A., Han, D., & Tacnet, J. M. (2020, July). 
        The SPOTIS rank reversal free method for multi-criteria decision-making support. 
        In 2020 IEEE 23rd International Conference on Information Fusion (FUSION) (pp. 1-8). 
        IEEE."""

        matrix = np.array([[15000, 4.3, 99, 42, 737],
                 [15290, 5.0, 116, 42, 892],
                 [15350, 5.0, 114, 45, 952],
                 [15490, 5.3, 123, 45, 1120]])

        weights = np.array([0.2941, 0.2353, 0.2353, 0.0588, 0.1765])
        types = np.array([-1, -1, -1, 1, 1])
        bounds_min = np.array([14000, 3, 80, 35, 650])
        bounds_max = np.array([16000, 8, 140, 60, 1300])
        bounds = np.vstack((bounds_min, bounds_max))


        method = SPOTIS()
        test_result = method(matrix, weights, types, bounds)
        
        real_result = np.array([0.4779, 0.5781, 0.5558, 0.5801])
        self.assertEqual(list(np.round(test_result, 3)), list(np.round(real_result, 3)))


# Test for the EDAS method
class Test_EDAS(unittest.TestCase):

    def test_edas(self):
        """Ersoy, Y. (2021). Equipment selection for an e-commerce company using entropy-based topsis, 
        edas and codas methods during the COVID-19. LogForum, 17(3)."""

        matrix = np.array([[256, 8, 41, 1.6, 1.77, 7347.16],
        [256, 8, 32, 1.0, 1.8, 6919.99],
        [256, 8, 53, 1.6, 1.9, 8400],
        [256, 8, 41, 1.0, 1.75, 6808.9],
        [512, 8, 35, 1.6, 1.7, 8479.99],
        [256, 4, 35, 1.6, 1.7, 7499.99]])

        weights = np.array([0.405, 0.221, 0.134, 0.199, 0.007, 0.034])
        types = np.array([1, 1, 1, 1, -1, -1])

        method = EDAS()
        test_result = method(matrix, weights, types)
        real_result = np.array([0.414, 0.130, 0.461, 0.212, 0.944, 0.043])
        self.assertEqual(list(np.round(test_result, 3)), list(np.round(real_result, 3)))


# Test for the MABAC method
class Test_MABAC(unittest.TestCase):

    def test_mabac(self):
        """Isik, O., Aydin, Y., & Kosaroglu, S. M. (2020). The assessment of the logistics 
        Performance Index of CEE Countries with the New Combination of SV and MABAC Methods. 
        LogForum, 16(4)."""

        matrix = np.array([[2.937588, 2.762986, 3.233723, 2.881315, 3.015289, 3.313491],
        [2.978555, 3.012820, 2.929487, 3.096154, 3.012820, 3.593939],
        [3.286673, 3.464600, 3.746009, 3.715632, 3.703427, 4.133620],
        [3.322037, 3.098638, 3.262154, 3.147851, 3.206675, 3.798684],
        [3.354866, 3.270945, 3.221880, 3.213207, 3.670508, 3.785941],
        [2.796570, 2.983000, 2.744904, 2.692550, 2.787563, 2.878851],
        [2.846491, 2.729618, 2.789990, 2.955624, 3.123323, 3.646595],
        [3.253458, 3.208902, 3.678499, 3.580044, 3.505663, 3.954262],
        [2.580718, 2.906903, 3.176497, 3.073653, 3.264727, 3.681887],
        [2.789011, 3.000000, 3.101099, 3.139194, 2.985348, 3.139194],
        [3.418681, 3.261905, 3.187912, 3.052381, 3.266667, 3.695238]])

        weights = np.array([0.171761, 0.105975, 0.191793, 0.168824, 0.161768, 0.199880])
        types = np.array([1, 1, 1, 1, 1, 1])

        method = MABAC(normalization_method=norms.minmax_normalization)
        test_result = method(matrix, weights, types)
        real_result = np.array([-0.155317, -0.089493, 0.505407, 0.132405, 0.246943, -0.386756, 
        -0.179406, 0.362921, -0.084198, -0.167505, 0.139895])
        
        self.assertEqual(list(np.round(test_result, 4)), list(np.round(real_result, 4)))


# Test for the MULTIMOORA method
class Test_MULTIMOORA(unittest.TestCase):

    def test_multimoora(self):
        """Karabasevic, D., Stanujkic, D., Urosevic, S., & Maksimovic, M. (2015). Selection of 
        candidates in the mining industry based on the application of the SWARA and the MULTIMOORA 
        methods. Acta Montanistica Slovaca, 20(2)."""

        matrix = np.array([[4, 3, 3, 4, 3, 2, 4],
        [3, 3, 4, 3, 5, 4, 4],
        [5, 4, 4, 5, 5, 5, 4]])

        weights = np.array([0.215, 0.215, 0.159, 0.133, 0.102, 0.102, 0.073])
        types = np.array([1, 1, 1, 1, 1, 1, 1])

        # MULTIMOORA RS
        method = MULTIMOORA_RS()
        test_result = method(matrix, weights, types)
        real_result = np.array([0.494, 0.527, 0.677])

        self.assertEqual(list(np.round(test_result, 3)), list(np.round(real_result, 3)))

        # MULTIMOORA RP
        method = MULTIMOORA_RP()
        test_result = method(matrix, weights, types)
        real_result = np.array([0.046, 0.061, 0.000])
        
        self.assertEqual(list(np.round(test_result, 3)), list(np.round(real_result, 3)))
        # MULTIMOORA FMF
        method = MULTIMOORA_FMF()
        test_result = method(matrix, weights, types)
        real_result = np.array([3.876e-09, 9.689e-09, 4.486e-08])
        
        self.assertEqual(list(np.round(test_result, 9)), list(np.round(real_result, 9)))

        # MULTIMOORA and `dominance_directed_graph` as compromise ranking
        mmoora = MULTIMOORA(compromise_rank_method=compromises.dominance_directed_graph)
        test_result = mmoora(matrix, weights, types)
        real_result = np.array([3, 2, 1])
        
        self.assertEqual(list(test_result), list(real_result))


# Test for borda copeland compromise ranking
class Test_Copeland(unittest.TestCase):

    def test_copeland(self):
        """Ecer, F. (2021). A consolidated MCDM framework for performance assessment of 
        battery electric vehicles based on ranking strategies. Renewable and Sustainable 
        Energy Reviews, 143, 110916."""

        matrix = np.array([[7, 8, 7, 6, 7, 7],
        [4, 7, 5, 7, 5, 4],
        [8, 9, 8, 8, 9, 8],
        [1, 4, 1, 1, 1, 1],
        [2, 2, 2, 4, 3, 2],
        [3, 1, 4, 3, 2, 3],
        [10, 5, 10, 9, 8, 10],
        [6, 3, 6, 5, 4, 6],
        [9, 10, 9, 10, 10, 9],
        [5, 6, 3, 2, 6, 5]])

        test_result = compromises.borda_copeland_compromise_ranking(matrix)
        real_result = np.array([7, 6, 8, 1, 2, 3, 9, 5, 10, 4])

        self.assertEqual(list(test_result), list(real_result))


# Test for dominance directed graph compromise ranking
class Test_Dominance_Directed_Graph(unittest.TestCase):

    def test_dominance_directed_graph(self):
        """Karabasevic, D., Stanujkic, D., Urosevic, S., & Maksimovic, M. (2015). Selection of 
        candidates in the mining industry based on the application of the SWARA and the 
        MULTIMOORA methods. Acta Montanistica Slovaca, 20(2)."""

        matrix = np.array([[3, 2, 3],
        [2, 3, 2],
        [1, 1, 1]])

        test_result = compromises.dominance_directed_graph(matrix)
        real_result = np.array([3, 2, 1])

        self.assertEqual(list(test_result), list(real_result))


# Test for rank preferences
class Test_Rank_preferences(unittest.TestCase):

    def test_rank_preferences(self):
        """Test based on paper Papathanasiou, J., & Ploskas, N. (2018). Vikor. In Multiple Criteria Decision Aid 
        (pp. 31-55). Springer, Cham."""

        pref = np.array([0.640, 1.000, 0.693, 0.271, 0.000, 0.694])
        test_result =rank_preferences(pref , reverse = False)
        real_result = np.array([3, 6, 4, 2, 1, 5])
        self.assertEqual(list(test_result), list(real_result))


def main():
    test_vikor = Test_VIKOR()
    test_vikor.test_vikor()

    test_rank_preferences = Test_Rank_preferences()
    test_rank_preferences.test_rank_preferences()

    test_topsis = Test_TOPSIS()
    test_topsis.test_topsis()

    test_codas = Test_CODAS()
    test_codas.test_codas()
    test_codas.test_codas2()

    test_waspas = Test_WASPAS()
    test_waspas.test_waspas()

    test_spotis = Test_SPOTIS()
    test_spotis.test_spotis()

    test_edas = Test_EDAS()
    test_edas.test_edas()

    test_mabac = Test_MABAC()
    test_mabac.test_mabac()

    test_multimoora = Test_MULTIMOORA()
    test_multimoora.test_multimoora()

    test_copeland = Test_Copeland()
    test_copeland.test_copeland()

    test_dominance_directed_graph = Test_Dominance_Directed_Graph()
    test_dominance_directed_graph.test_dominance_directed_graph()


if __name__ == '__main__':
    main()