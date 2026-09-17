import unittest

import numpy as np

from backpropagation import Value, branched_expression, scalar_chain_rule
from optimization import eval_numerical_gradient


class ValueTests(unittest.TestCase):
    def test_automatic_gradients_match_the_manual_chain_rule(self):
        x, y, z = Value(-2), Value(5), Value(-4)
        output = (x + y) * z
        output.backward()
        expected, gradients = scalar_chain_rule(x.data, y.data, z.data)
        self.assertEqual(output.data, expected)
        self.assertEqual({"x": x.grad, "y": y.grad, "z": z.grad}, gradients)
        self.assertEqual(output.grad, 1.0)

    def test_shared_inputs_accumulate_gradients_from_every_branch(self):
        x, y = Value(2), Value(-3)
        output = x * y + (x + y) ** 2
        output.backward()
        expected, gradients = branched_expression(x.data, y.data)
        self.assertEqual(output.data, expected)
        self.assertEqual({"x": x.grad, "y": y.grad}, gradients)

    def test_repeated_edges_and_shared_intermediate_are_counted_correctly(self):
        x = Value(3)
        square = x * x
        output = square * square + square  # x**4 + x**2
        output.backward()
        self.assertEqual(square.grad, 19.0)
        self.assertEqual(x.grad, 114.0)

    def test_composite_arithmetic_matches_numerical_gradients(self):
        def expression(x, y):
            return (2 + x) * (3 - y) / (x + 4) + 5 / y - x / 2 + x ** 2.5 - y

        inputs = np.array([1.2, 0.7])
        x, y = Value(inputs[0]), Value(inputs[1])
        output = expression(x, y)
        output.backward()
        numerical = eval_numerical_gradient(lambda a: expression(*a), inputs)
        self.assertAlmostEqual(output.data, expression(*inputs))
        np.testing.assert_allclose([x.grad, y.grad], numerical, atol=1e-8)

    def test_sigmoid_neuron_matches_numerical_gradients(self):
        parameters = np.array([0.2, -0.3, 0.1])
        w0, w1, b = [Value(item) for item in parameters]
        output = (w0 * 2 + w1 * -1 + b).sigmoid()
        output.backward()
        numerical = eval_numerical_gradient(
            lambda p: 1 / (1 + np.exp(-(p[0] * 2 - p[1] + p[2]))), parameters,
        )
        np.testing.assert_allclose([w0.grad, w1.grad, b.grad], numerical, atol=1e-8)

    def test_exp_log_and_relu_match_numerical_gradients(self):
        x = Value(0.7)
        output = x.exp().log() + (x * 2).relu() + (-x).relu()
        output.backward()
        numerical = eval_numerical_gradient(
            lambda a: np.log(np.exp(a)) + np.maximum(0, a * 2) + np.maximum(0, -a),
            np.array(0.7),
        )
        self.assertAlmostEqual(x.grad, float(numerical))

    def test_repeated_backward_resets_gradients_and_accepts_an_upstream_seed(self):
        x = Value(2)
        square = x * x
        output = square * 3
        for seed in (1.0, 1.0, 0.5):
            output.backward(seed)
            self.assertEqual(x.grad, 12 * seed)
            self.assertEqual(square.grad, 3 * seed)
        x.backward()
        self.assertEqual(x.grad, 1.0)

    def test_backward_on_a_different_output_resets_shared_leaf_gradients(self):
        x = Value(2)
        (x ** 2).backward()
        self.assertEqual(x.grad, 4.0)
        (x ** 3).backward()
        self.assertEqual(x.grad, 12.0)

    def test_forward_derivatives_are_cached(self):
        x = Value(2)
        output = x * x + x ** 2
        x.data = 10
        output.backward()
        self.assertEqual(output.data, 8.0)
        self.assertEqual(x.grad, 8.0)  # Derivative at the forward-pass value, 2.

    def test_long_graph_does_not_require_recursive_traversal(self):
        x = Value(1)
        output = x
        for _ in range(3000):
            output = output + 1
        output.backward()
        self.assertEqual(output.data, 3001)
        self.assertEqual(x.grad, 1)

    def test_activation_boundaries_and_constant_power(self):
        for data, expected_output, expected_grad in ((-1000, 0, 0), (0, 0.5, 0.25),
                                                     (1000, 1, 0)):
            x = Value(data)
            output = x.sigmoid()
            output.backward()
            self.assertEqual(output.data, expected_output)
            self.assertEqual(x.grad, expected_grad)
        zero = Value(0)
        zero.relu().backward()
        self.assertEqual(zero.grad, 0)
        constant = zero ** 0
        constant.backward()
        self.assertEqual(constant.data, 1)
        self.assertEqual(zero.grad, 0)

    def test_unsupported_types_and_undefined_operations_fail_clearly(self):
        with self.assertRaises(TypeError):
            Value(np.ones(2))
        with self.assertRaises(TypeError):
            Value(2) ** Value(3)
        with self.assertRaises(ZeroDivisionError):
            Value(1) / 0
        with self.assertRaises(ValueError):
            Value(-1).log()
        with self.assertRaises(ValueError):
            Value(-1) ** 0.5
        with self.assertRaises(ValueError):
            Value(0) ** 0.5

    def test_gradient_descent_rebuilds_the_graph_and_reduces_loss(self):
        weight = Value(0)
        for _ in range(20):
            loss = (weight - 3) ** 2
            loss.backward()
            weight.data -= 0.1 * weight.grad
        self.assertLess((weight.data - 3) ** 2, 0.01)


if __name__ == "__main__":
    unittest.main()
