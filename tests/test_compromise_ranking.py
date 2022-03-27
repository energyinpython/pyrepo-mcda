from pyrepo_mcda import compromise_rankings as compromises
from pyrepo_mcda.additions import rank_preferences

import unittest
import numpy as np

# Test for Borda Copeland compromise ranking
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

        test_result = compromises.copeland(matrix)
        real_result = np.array([7, 6, 8, 1, 2, 3, 9, 5, 10, 4])

        self.assertEqual(list(test_result), list(real_result))


# Test for Dominance Directed Graph compromise ranking
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

    def test_dominance_directed_graph2(self):
        """Test based on Altuntas, S., Dereli, T., & Yilmaz, M. K. (2015). Evaluation of 
        excavator technologies: application of data fusion based MULTIMOORA methods. Journal 
        of Civil Engineering and Management, 21(8), 977-997."""
        matrix = np.array([[36, 36, 28],
        [28, 45, 45],
        [80, 66, 78],
        [0, 3, 1],
        [6, 10, 3],
        [1, 1, 0],
        [3, 0, 6],
        [68, 78, 55],
        [56, 55, 36],
        [15, 15, 10],
        [68, 28, 66],
        [10, 6, 15],
        [21, 21, 21]])

        for j in range(matrix.shape[1]):
            matrix[:, j] = rank_preferences(matrix[:, j], reverse = True)

        test_result = compromises.dominance_directed_graph(matrix)
        real_result = np.array([6, 5, 1, 12, 10, 13, 11, 2, 4, 8, 3, 9, 7])

        self.assertEqual(list(test_result), list(real_result))


# Test for Rank Position Method compromise ranking
class Test_Rank_Position_Method(unittest.TestCase):

    def test_rank_position_method(self):
        """Test based on Altuntas, S., Dereli, T., & Yilmaz, M. K. (2015). Evaluation of 
        excavator technologies: application of data fusion based MULTIMOORA methods. Journal 
        of Civil Engineering and Management, 21(8), 977-997."""
        matrix = np.array([[1, 2],
        [2, 3],
        [4, 4],
        [3, 1]])

        test_result = compromises.rank_position_method(matrix)
        real_result = np.array([1, 3, 4, 2])
        self.assertEqual(list(test_result), list(real_result))

    def test_rank_position_method2(self):
        """Test based on Altuntas, S., Dereli, T., & Yilmaz, M. K. (2015). Evaluation of 
        excavator technologies: application of data fusion based MULTIMOORA methods. Journal 
        of Civil Engineering and Management, 21(8), 977-997."""
        matrix = np.array([[36, 36, 28],
        [28, 45, 45],
        [80, 66, 78],
        [0, 3, 1],
        [6, 10, 3],
        [1, 1, 0],
        [3, 0, 6],
        [68, 78, 55],
        [56, 55, 36],
        [15, 15, 10],
        [68, 28, 66],
        [10, 6, 15],
        [21, 21, 21]])

        for j in range(matrix.shape[1]):
            matrix[:, j] = rank_preferences(matrix[:, j], reverse = True)

        test_result = compromises.rank_position_method(matrix)
        real_result = np.array([6, 5, 1, 12, 10, 13, 11, 2, 4, 8, 3, 9, 7])

        self.assertEqual(list(test_result), list(real_result))


def main():

    test_copeland = Test_Copeland()
    test_copeland.test_copeland()

    test_dominance_directed_graph = Test_Dominance_Directed_Graph()
    test_dominance_directed_graph.test_dominance_directed_graph()
    test_dominance_directed_graph.test_dominance_directed_graph2()

    test_rank_position_method = Test_Rank_Position_Method()
    test_rank_position_method.test_rank_position_method()
    test_rank_position_method.test_rank_position_method2()


if __name__ == '__main__':
    main()