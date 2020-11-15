from gauss_jordan_elimination import pivot, execute
import numpy as np
import unittest


class TestGaussJordanElimination(unittest.TestCase):
    def test_pivot(self):
        A = np.array(
            [
                [10.0, 11.0, 4.0, 0.0],
                [1.0, 10.0, 5.0, -1.0],
                [4.0, 5.0, 10.0, 7.0],
                [0.0, -1.0, 7.0, 9.0],
            ],
        )
        A_pivot = np.array(
            [
                [11.0, 10.0, 4.0, 0.0],
                [10.0, 1.0, 5.0, -1.0],
                [5.0, 4.0, 10.0, 7.0],
                [-1.0, 0.0, 7.0, 9.0],
            ],
        )

        pivot(A, 0)
        self.assertTrue(np.array_equal(A, A_pivot))

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
        self.assertTrue((np.abs(b - A @ x.T) < 0.001).all())


if __name__ == "__main__":
    unittest.main()