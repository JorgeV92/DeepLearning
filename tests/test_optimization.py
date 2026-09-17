import unittest

import numpy as np

from lc import softmax_loss, svm_loss
from optimization import (
    eval_numerical_gradient,
    gradient_descent,
    stochastic_gradient_descent,
)


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2.0, -1.0], [-1.0, -2.0],
                           [1.0, 2.0], [2.0, 1.0]])
        self.y = np.array([0, 0, 1, 1])
        self.W = np.zeros((2, 2))
        self.b = np.zeros(2)

    def test_numerical_gradient_of_quadratic_preserves_input(self):
        x = np.array([[1, -2], [3, 0]], dtype=np.float32)
        original = x.copy()
        grad = eval_numerical_gradient(lambda value: np.sum(value ** 2), x)
        np.testing.assert_allclose(grad, 2 * x, atol=1e-8)
        np.testing.assert_array_equal(x, original)

    def test_linear_loss_gradients_match_finite_differences(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(8, 3))
        y = np.array([0, 1, 2, 0, 1, 0, 1, 0])
        W = rng.normal(scale=0.1, size=(3, 3))
        b = rng.normal(scale=0.1, size=3)
        for loss_fn in (svm_loss, softmax_loss):
            with self.subTest(loss=loss_fn.__name__):
                _, dW, db = loss_fn(W, b, X, y, reg=0.2)
                numerical_dW = eval_numerical_gradient(
                    lambda value: loss_fn(value, b, X, y, 0.2)[0], W
                )
                numerical_db = eval_numerical_gradient(
                    lambda value: loss_fn(W, value, X, y, 0.2)[0], b
                )
                np.testing.assert_allclose(dW, numerical_dW, atol=1e-8, rtol=1e-6)
                np.testing.assert_allclose(db, numerical_db, atol=1e-8, rtol=1e-6)

    def test_optimizers_learn_separable_data_and_preserve_starting_parameters(self):
        for loss_fn in (svm_loss, softmax_loss):
            for optimizer in (gradient_descent, stochastic_gradient_descent):
                with self.subTest(loss=loss_fn.__name__, optimizer=optimizer.__name__):
                    initial_loss = loss_fn(self.W, self.b, self.X, self.y, 0.01)[0]
                    W, b, history = optimizer(
                        loss_fn, self.W, self.b, self.X, self.y,
                        learning_rate=0.1, reg=0.01, num_iters=40,
                    )
                    final_loss = loss_fn(W, b, self.X, self.y, 0.01)[0]
                    self.assertLess(final_loss, initial_loss)
                    np.testing.assert_array_equal(np.argmax(self.X @ W + b, axis=1), self.y)
                    np.testing.assert_array_equal(self.W, np.zeros((2, 2)))
                    np.testing.assert_array_equal(self.b, np.zeros(2))
                    self.assertEqual(len(history), 40)

    def test_full_size_sgd_batches_agree_with_gradient_descent(self):
        W_gd, b_gd, history_gd = gradient_descent(
            softmax_loss, self.W, self.b, self.X, self.y, num_iters=5
        )
        for batch_size in (len(self.X), len(self.X) + 5):
            with self.subTest(batch_size=batch_size):
                W_sgd, b_sgd, history_sgd = stochastic_gradient_descent(
                    softmax_loss, self.W, self.b, self.X, self.y,
                    num_iters=5, batch_size=batch_size,
                )
                np.testing.assert_allclose(W_sgd, W_gd, atol=1e-12)
                np.testing.assert_allclose(b_sgd, b_gd, atol=1e-12)
                np.testing.assert_allclose(history_sgd, history_gd)

    def test_sgd_is_reproducible_and_uses_requested_batch_size(self):
        for batch_size in (1, 3):
            batches = []

            def record_loss(W, b, X, y, reg):
                batches.append(X.copy())
                return softmax_loss(W, b, X, y, reg)

            first = stochastic_gradient_descent(
                record_loss, self.W, self.b, self.X, self.y,
                num_iters=5, batch_size=batch_size, seed=123,
            )
            second = stochastic_gradient_descent(
                softmax_loss, self.W, self.b, self.X, self.y,
                num_iters=5, batch_size=batch_size, seed=123,
            )
            self.assertEqual([len(batch) for batch in batches], [batch_size] * 5)
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a, b)

    def test_gradient_descent_updates_bias_and_records_pre_update_loss(self):
        X = np.zeros((3, 2))
        y = np.zeros(3, dtype=int)
        W, b, history = gradient_descent(
            softmax_loss, self.W, self.b, X, y,
            learning_rate=0.1, reg=0, num_iters=1,
        )
        np.testing.assert_array_equal(W, self.W)
        np.testing.assert_allclose(b, [0.05, -0.05])
        np.testing.assert_allclose(history, [np.log(2)])


if __name__ == "__main__":
    unittest.main()
