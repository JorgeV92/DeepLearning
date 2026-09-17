import unittest

import numpy as np

from backpropagation import (
    affine_backward,
    affine_forward,
    branched_expression,
    linear_softmax_loss,
    relu_backward,
    relu_forward,
    run_gradient_checks,
    scalar_chain_rule,
    sigmoid_backward,
    sigmoid_forward,
    softmax_cross_entropy,
)
from lc import softmax_loss
from optimization import eval_numerical_gradient, stochastic_gradient_descent


class BackpropagationTests(unittest.TestCase):
    def test_scalar_chain_and_branch_gradients(self):
        for function, inputs in (
            (scalar_chain_rule, np.array([-2.0, 5.0, -4.0])),
            (branched_expression, np.array([2.0, -3.0])),
        ):
            with self.subTest(function=function.__name__):
                _, gradients = function(*inputs)
                numerical = eval_numerical_gradient(lambda x: function(*x)[0], inputs)
                np.testing.assert_allclose(list(gradients.values()), numerical, atol=1e-8)

    def test_affine_gradients_match_finite_differences_with_arbitrary_upstream(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(3, 4))
        W = rng.normal(size=(4, 2))
        b = rng.normal(size=2)
        upstream = rng.normal(size=(3, 2))
        _, cache = affine_forward(X, W, b)
        dX, dW, db = affine_backward(upstream, cache)
        for parameter, actual, objective in (
            (X, dX, lambda x: np.sum(affine_forward(x, W, b)[0] * upstream)),
            (W, dW, lambda w: np.sum(affine_forward(X, w, b)[0] * upstream)),
            (b, db, lambda bias: np.sum(affine_forward(X, W, bias)[0] * upstream)),
        ):
            self.assertEqual(actual.shape, parameter.shape)
            np.testing.assert_allclose(
                actual, eval_numerical_gradient(objective, parameter), atol=1e-8,
            )

    def test_activation_gradients_match_finite_differences(self):
        # Stay away from zero for ReLU's finite-difference check.
        x = np.array([[-2.0, -0.3], [0.7, 3.0]])
        upstream = np.array([[2.0, -1.0], [0.5, -3.0]])
        for forward, backward in ((sigmoid_forward, sigmoid_backward),
                                  (relu_forward, relu_backward)):
            with self.subTest(activation=forward.__name__):
                out, cache = forward(x)
                original = out.copy()
                actual = backward(upstream, cache)
                numerical = eval_numerical_gradient(
                    lambda value: np.sum(forward(value)[0] * upstream), x,
                )
                np.testing.assert_allclose(actual, numerical, atol=1e-8)
                np.testing.assert_array_equal(out, original)

    def test_sigmoid_is_stable_for_large_inputs(self):
        with np.errstate(over="raise", invalid="raise"):
            out, cache = sigmoid_forward(np.array([-1000.0, 0.0, 1000.0]))
            grad = sigmoid_backward(np.ones(3), cache)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(grad, [0.0, 0.25, 0.0])

    def test_relu_routes_gradients_and_uses_zero_at_the_kink(self):
        out, cache = relu_forward(np.array([-2.0, 0.0, 3.0]))
        grad = relu_backward(np.array([7.0, 8.0, 9.0]), cache)
        np.testing.assert_array_equal(out, [0.0, 0.0, 3.0])
        np.testing.assert_array_equal(grad, [0.0, 0.0, 9.0])

    def test_softmax_score_gradient_and_shift_invariance(self):
        scores = np.array([[0.2, -0.1, 0.5], [-0.5, 0.3, 0.1]])
        y = np.array([1, 2])
        original = scores.copy()
        loss, grad = softmax_cross_entropy(scores, y)
        numerical = eval_numerical_gradient(lambda x: softmax_cross_entropy(x, y)[0], scores)
        np.testing.assert_allclose(grad, numerical, atol=1e-8)
        np.testing.assert_array_equal(scores, original)
        shifted_loss, shifted_grad = softmax_cross_entropy(scores + 10000, y)
        self.assertAlmostEqual(loss, shifted_loss)
        np.testing.assert_allclose(grad, shifted_grad, atol=1e-10)

    def test_softmax_loss_remains_finite_for_an_extremely_wrong_prediction(self):
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            loss, grad = softmax_cross_entropy(
                np.array([[1000.0, -1000.0]]), np.array([1]),
            )
        self.assertEqual(loss, 2000.0)
        np.testing.assert_allclose(grad, [[1.0, -1.0]])

    def test_modular_loss_matches_existing_classifier_including_regularization(self):
        rng = np.random.default_rng(17)
        X = rng.normal(size=(6, 4))
        W = rng.normal(scale=0.1, size=(4, 3))
        b = rng.normal(scale=0.1, size=3)
        y = np.array([0, 2, 1, 0, 1, 1])
        for reg in (0.0, 0.2):
            with self.subTest(reg=reg):
                actual = linear_softmax_loss(W, b, X, y, reg)
                expected = softmax_loss(W, b, X, y, reg)
                for a, e in zip(actual, expected):
                    np.testing.assert_allclose(a, e, atol=1e-12)

    def test_composed_neuron_and_loss_pass_gradient_checks(self):
        for name, error in run_gradient_checks().items():
            with self.subTest(name=name):
                self.assertLess(error, 1e-8)

    def test_backprop_gradients_train_using_the_existing_optimizer(self):
        X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
        y = np.array([0, 0, 1, 1])
        W = np.zeros((2, 2))
        b = np.zeros(2)
        initial_loss = linear_softmax_loss(W, b, X, y, 0.01)[0]
        W, b, history = stochastic_gradient_descent(
            linear_softmax_loss, W, b, X, y, learning_rate=0.1,
            reg=0.01, batch_size=2, num_iters=40,
        )
        final_loss = linear_softmax_loss(W, b, X, y, 0.01)[0]
        self.assertLess(final_loss, initial_loss)
        self.assertEqual(len(history), 40)
        np.testing.assert_array_equal(np.argmax(X @ W + b, axis=1), y)


if __name__ == "__main__":
    unittest.main()
