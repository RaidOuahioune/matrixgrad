import unittest
import numpy as np
from matrixgrad.engine import (
    Tensor,
)  # Replace 'your_module' with the actual module name


class TestMathFunctions(unittest.TestCase):

    def test_add(self):

        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        result = t1 + t2
        expected_result = Tensor([5, 7, 9])
        np.testing.assert_array_equal(result.data, expected_result.data)

        t1 = Tensor([1, 2, 3])
        result = t1 + 4
        expected_result = Tensor([5, 6, 7])
        np.testing.assert_array_equal(result.data, expected_result.data)

    def test_mul(self):

        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        result = t1 * t2
        expected_result = Tensor([4, 10, 18])
        np.testing.assert_array_equal(result.data, expected_result.data)

        t1 = Tensor([1, 2, 3])
        result = t1 * 4
        expected_result = Tensor([4, 8, 12])
        np.testing.assert_array_equal(result.data, expected_result.data)

    def test_matmul(self):

        t1 = Tensor([[1, 2], [3, 4]])
        t2 = Tensor([[5, 6], [7, 8]])
        result = t1 @ t2
        expected_result = Tensor([[19, 22], [43, 50]])
        np.testing.assert_array_equal(result.data, expected_result.data)

    def test_pow(self):

        t1 = Tensor([1, 2, 3])
        result = t1**2
        expected_result = Tensor([1, 4, 9])
        np.testing.assert_array_equal(result.data, expected_result.data)

    def test_relu(self):

        t1 = Tensor([-1, 0, 1])
        result = t1.relu()
        expected_result = Tensor([0, 0, 1])
        np.testing.assert_array_equal(result.data, expected_result.data)

    def test_sum(self):
        t1 = Tensor([1, 2, 3])
        result = t1.sum()
        expected_result = Tensor([6])
        np.testing.assert_array_equal(result.data, expected_result.data)


if __name__ == "__main__":
    unittest.main()
