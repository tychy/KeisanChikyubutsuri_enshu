from sor import split_matrix, execute
import numpy as np
import unittest


class TestGaussJordanElimination(unittest.TestCase):
    def test_split_matrix(self):
        A = np.array(
            [
                [1.0, 11.0, 4.0, 0.0],
                [10.0, 10.0, 5.0, -1.0],
                [4.0, 5.0, 10.0, 7.0],
                [0.0, -1.0, 7.0, 9.0],
            ],
        )
        D_true = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0, 9.0],
            ],
        )
        L_true = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0],
                [4.0, 5.0, 0.0, 0.0],
                [0.0, -1.0, 7.0, 0.0],
            ],
        )
        U_true = np.array(
            [
                [0.0, 11.0, 4.0, 0.0],
                [0.0, 0.0, 5.0, -1.0],
                [0.0, 0.0, 0.0, 7.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        )
        D, L, U = split_matrix(A)
        self.assertTrue(np.array_equal(D, D_true))
        self.assertTrue(np.array_equal(L, L_true))
        self.assertTrue(np.array_equal(U, U_true))

    def test_execute(self):
        A = np.array(
            [
                [10.0, 1.0, 4.0, 0.0],
                [1.0, 10.0, 5.0, -1.0],
                [4.0, 5.0, 10.0, 7.0],
                [0.0, -1.0, 7.0, 9.0],
            ],
        )
        b = np.array([15.0, 15.0, 25.0, 15.0])
        x = execute(np.copy(A), np.copy(b))
        print(A)
        print(x)
        print(A @ x.T)
        self.assertTrue((np.abs(b - A @ x.T) < 0.001).all())


if __name__ == "__main__":
    unittest.main()
