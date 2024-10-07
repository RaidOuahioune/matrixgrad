import unittest
import numpy as np
from engine import Tensor


class TestBackwardFunctions(unittest.TestCase):

    def test_add_backward(self):
        # Test backward function for addition
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        result = t1 + t2
        result.backward()
        expected_grad_t1 = np.array([[1, 1, 1]], dtype=np.float64)
        expected_grad_t2 = np.array([[1, 1, 1]], dtype=np.float64)
        np.testing.assert_array_equal(t1.grad, expected_grad_t1)
        np.testing.assert_array_equal(t2.grad, expected_grad_t2)

    def test_mul_backward(self):
        # Test backward function for multiplication
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        result = t1 * t2
        result.backward()
        print(t1.grad)

        expected_grad_t1 = np.array([[4, 5, 6]], dtype=np.float64)
        expected_grad_t2 = np.array([[1, 2, 3]], dtype=np.float64)
        np.testing.assert_array_equal(t1.grad, expected_grad_t1)
        np.testing.assert_array_equal(t2.grad, expected_grad_t2)

    def test_matmul_backward(self):
        # Test backward function for matrix multiplication
        t1 = Tensor([[1, 2], [3, 4]])  # A matrix of shape (2, 2)
        t2 = Tensor([[5, 6], [7, 8]])  # B matrix of shape (2, 2)

        result = t1 @ t2
        result.backward()  # Assume gradient propagation of ones for simplicity

        # Gradients for t1 and t2 will be based on each other's transpose
        expected_grad_t1 = np.array(
            [[5 + 6, 7 + 8], [5 + 6, 7 + 8]], dtype=np.float64
        )  # B's transpose times ones
        expected_grad_t2 = np.array(
            [[1 + 3, 1 + 3], [2 + 4, 2 + 4]], dtype=np.float64
        )  # A's transpose times ones

        np.testing.assert_array_equal(t1.grad, expected_grad_t1)
        np.testing.assert_array_equal(t2.grad, expected_grad_t2)

    def test_pow_backward(self):
        # Test backward function for exponentiation
        t1 = Tensor([1, 2, 3])
        result = t1**2
        result.backward()
        expected_grad_t1 = np.array([[2, 4, 6]], dtype=np.float64)
        np.testing.assert_array_equal(t1.grad, expected_grad_t1)

    def test_relu_backward(self):
        # Test backward function for ReLU activation function
        t1 = Tensor([-1, 0, 1])
        result = t1.relu()
        result.backward()
        expected_grad_t1 = np.array([[0, 0, 1]], dtype=np.float64)
        np.testing.assert_array_equal(t1.grad, expected_grad_t1)

    def test_sum_backward(self):
        # Test backward function for sum
        t1 = Tensor([1, 2, 3])
        result = t1.sum()
        result.backward()
        expected_grad_t1 = np.array([[1, 1, 1]], dtype=np.float64)
        np.testing.assert_array_equal(t1.grad, expected_grad_t1)


if __name__ == "__main__":
    unittest.main()
