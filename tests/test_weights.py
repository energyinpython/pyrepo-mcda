import unittest
import numpy as np
from pyrepo_mcda import weighting_methods as mcda_weights


# Test for CRITIC weighting
class Test_CRITIC(unittest.TestCase):

    def test_critic(self):
        """Test based on paper Tuş, A., & Aytaç Adalı, E. (2019). The new combination with CRITIC and WASPAS methods 
        for the time and attendance software selection problem. Opsearch, 56(2), 528-538."""

        matrix = np.array([[5000, 3, 3, 4, 3, 2],
        [680, 5, 3, 2, 2, 1],
        [2000, 3, 2, 3, 4, 3],
        [600, 4, 3, 1, 2, 2],
        [800, 2, 4, 3, 3, 4]])

        types = np.array([-1, 1, 1, 1, 1, 1])

        test_result = mcda_weights.critic_weighting(matrix)
        real_result = np.array([0.157, 0.249, 0.168, 0.121, 0.154, 0.151])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for Entropy weighting
class Test_Entropy(unittest.TestCase):

    def test_Entropy(self):
        """Test based on paper Xu, X. (2004). A note on the subjective and objective integrated approach to 
        determine attribute weights. European Journal of Operational Research, 156(2), 
        530-532."""

        matrix = np.array([[30, 30, 38, 29],
        [19, 54, 86, 29],
        [19, 15, 85, 28.9],
        [68, 70, 60, 29]])

        types = np.array([1, 1, 1, 1])

        test_result = mcda_weights.entropy_weighting(matrix)
        real_result = np.array([0.4630, 0.3992, 0.1378, 0.0000])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))

    def test_Entropy2(self):
        """Test based on paper Zavadskas, E. K., & Podvezko, V. (2016). Integrated determination of objective 
        criteria weights in MCDM. International Journal of Information Technology & Decision 
        Making, 15(02), 267-283."""

        matrix = np.array([[3.0, 100, 10, 7],
        [2.5, 80, 8, 5],
        [1.8, 50, 20, 11],
        [2.2, 70, 12, 9]])

        types = np.array([-1, 1, -1, 1])

        test_result = mcda_weights.entropy_weighting(matrix)
        real_result = np.array([0.1146, 0.1981, 0.4185, 0.2689])
        self.assertEqual(list(np.round(test_result, 4)), list(real_result))


    def test_Entropy3(self):
        """Ersoy, Y. (2021). Equipment selection for an e-commerce company using entropy-based 
        topsis, edas and codas methods during the COVID-19. LogForum, 17(3)."""

        matrix = np.array([[256, 8, 41, 1.6, 1.77, 7347.16],
        [256, 8, 32, 1.0, 1.8, 6919.99],
        [256, 8, 53, 1.6, 1.9, 8400],
        [256, 8, 41, 1.0, 1.75, 6808.9],
        [512, 8, 35, 1.6, 1.7, 8479.99],
        [256, 4, 35, 1.6, 1.7, 7499.99]])

        types = np.array([-1, 1, -1, 1])
        test_result = mcda_weights.entropy_weighting(matrix)

        real_result = np.array([0.405, 0.221, 0.134, 0.199, 0.007, 0.034])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))

    def test_Entropy4(self):

        matrix = np.array([[4550, 30, 6.74, 20, 15, 5, 85, 150, 0.87, 4.76],
        [3005, 60.86, 2.4, 35, 27, 4, 26, 200, 0.17, 4.51],
        [2040, 14.85, 1.7, 90, 25, 5, 26, 500, 0.27, 4.19],
        [3370, 99.4, 3.25, 25.3, 54, 3, 45, 222, 0.21, 3.78],
        [3920, 112.6, 4.93, 11.4, 71.7, 2, 50, 100, 0.25, 4.11]])

        test_result = mcda_weights.entropy_weighting(matrix)
        real_result = np.array([0.026, 0.154, 0.089, 0.199, 0.115, 0.04, 0.08, 0.123, 0.172, 0.002])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))


# Test for Standard Deviation weighting
class Test_STD(unittest.TestCase):

    def test_std(self):
        """Test based on paper Sałabun, W., Wątróbski, J., & Shekhovtsov, A. (2020). Are mcda methods benchmarkable? 
        a comparative study of topsis, vikor, copras, and promethee ii methods. Symmetry, 12(9), 
        1549."""

        matrix = np.array([[0.619, 0.449, 0.447],
        [0.862, 0.466, 0.006],
        [0.458, 0.698, 0.771],
        [0.777, 0.631, 0.491],
        [0.567, 0.992, 0.968]])
        
        types = np.array([1, 1, 1])

        test_result = mcda_weights.std_weighting(matrix)
        real_result = np.array([0.217, 0.294, 0.488])
        self.assertEqual(list(np.round(test_result, 3)), list(real_result))
        


def main():
    test_critic = Test_CRITIC()
    test_critic.test_critic()

    test_entropy = Test_Entropy()
    test_entropy.test_Entropy()
    test_entropy.test_Entropy2()
    test_entropy.test_Entropy3()
    test_entropy.test_Entropy4()

    test_std = Test_STD()
    test_std.test_std()
    

if __name__ == '__main__':
    main()

