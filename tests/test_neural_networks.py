import unittest

import numpy as np

from lc import softmax_loss
from neural_networks import MLP, run_gradient_checks


class MLPTests(unittest.TestCase):
    def test_multiple_hidden_layers_match_a_hand_computed_forward_pass(self):
        model = MLP(2, hidden_dims=(2, 2), num_classes=2, dtype=np.float64)
        model.params = {
            "W1": np.array([[1.0, -1.0], [2.0, 1.0]]),
            "b1": np.array([-1.0, 0.5]),
            "W2": np.array([[1.0, -2.0], [0.5, 1.0]]),
            "b2": np.array([0.1, 0.2]),
            "W3": np.array([[-1.0, 2.0], [1.0, -1.0]]),
            "b3": np.array([-0.5, -0.2]),
        }
        X = np.array([[1.0, -2.0], [2.0, 1.0]])
        scores, _ = model.forward(X)
        # Negative output scores must survive: ReLU applies only to hidden layers.
        np.testing.assert_allclose(scores, [[-0.4, -0.2], [-3.6, 6.0]])
        np.testing.assert_array_equal(model.pred(X), [1, 1])

    def test_every_parameter_gradient_matches_finite_differences(self):
        errors = run_gradient_checks()
        self.assertEqual(set(errors), {"W1", "b1", "W2", "b2", "W3", "b3"})
        for name, error in errors.items():
            with self.subTest(parameter=name):
                self.assertLess(error, 1e-8)

    def test_no_hidden_layers_reproduce_the_existing_linear_classifier(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(5, 3))
        y = np.array([0, 1, 2, 1, 0])
        model = MLP(3, hidden_dims=(), num_classes=3, dtype=np.float64)
        loss, gradients = model.loss(X, y, reg=0.2)
        expected_loss, dW, db = softmax_loss(model.params["W1"], model.params["b1"], X, y, 0.2)
        self.assertAlmostEqual(loss, expected_loss)
        np.testing.assert_allclose(gradients["W1"], dW, atol=1e-12)
        np.testing.assert_allclose(gradients["b1"], db, atol=1e-12)

    def test_regularization_applies_to_all_weight_matrices_and_excludes_biases(self):
        rng = np.random.default_rng(5)
        model = MLP(3, hidden_dims=(4, 3), num_classes=2, dtype=np.float64)
        # Nonzero biases make accidental bias regularization detectable.
        for name, parameter in model.params.items():
            if name.startswith("b"):
                parameter[:] = 0.2
        X = rng.normal(size=(5, 3))
        y = np.array([0, 1, 1, 0, 1])
        unregularized_loss, unregularized_gradients = model.loss(X, y, reg=0)
        loss, gradients = model.loss(X, y, reg=0.3)
        expected_penalty = 0.3 * sum(np.sum(p ** 2) for name, p in model.params.items()
                                     if name.startswith("W"))
        self.assertAlmostEqual(loss - unregularized_loss, expected_penalty)
        for name, parameter in model.params.items():
            expected_change = 0.6 * parameter if name.startswith("W") else np.zeros_like(parameter)
            np.testing.assert_allclose(gradients[name] - unregularized_gradients[name],
                                       expected_change, atol=1e-12)

    def test_loss_preserves_inputs_and_parameters_and_matches_gradient_shapes(self):
        X = np.array([[0.3, -0.4], [0.5, 0.2]], dtype=np.float32)
        y = np.array([0, 1])
        model = MLP(2, hidden_dims=(3, 4), num_classes=2)
        original_X = X.copy()
        original_params = {name: parameter.copy() for name, parameter in model.params.items()}
        _, gradients = model.loss(X, y)
        np.testing.assert_array_equal(X, original_X)
        self.assertEqual(set(gradients), set(model.params))
        for name, parameter in model.params.items():
            np.testing.assert_array_equal(parameter, original_params[name])
            self.assertEqual(gradients[name].shape, parameter.shape)

    def test_mlp_learns_xor_and_updates_every_layer(self):
        X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
        y = np.array([0, 1, 1, 0])
        model = MLP(2, hidden_dims=(8,), num_classes=2)
        original_params = {name: parameter.copy() for name, parameter in model.params.items()}
        initial_loss = model.loss(X, y, reg=0)[0]
        history = model.train(X, y, learning_rate=0.1, reg=0, num_iters=500, batch_size=4)
        self.assertEqual(len(history), 500)
        self.assertLess(model.loss(X, y, reg=0)[0], 0.1 * initial_loss)
        np.testing.assert_array_equal(model.pred(X), y)
        self.assertEqual(model.accuracy(X, y), 1.0)
        for name, parameter in model.params.items():
            self.assertFalse(np.array_equal(parameter, original_params[name]), name)

    def test_initialization_and_sampled_training_are_reproducible(self):
        X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
        y = np.array([0, 1, 1, 0])
        first = MLP(2, hidden_dims=(4, 3), num_classes=2, seed=10)
        second = MLP(2, hidden_dims=(4, 3), num_classes=2, seed=10)
        history_a = first.train(X, y, num_iters=5, batch_size=2, seed=11)
        history_b = second.train(X, y, num_iters=5, batch_size=2, seed=11)
        np.testing.assert_array_equal(history_a, history_b)
        for name in first.params:
            np.testing.assert_array_equal(first.params[name], second.params[name])

    def test_invalid_data_and_training_settings_fail_before_updating_parameters(self):
        model = MLP(2, hidden_dims=(3,), num_classes=2)
        X = np.ones((3, 2))
        y = np.array([0, 1, 0])
        original_params = {name: parameter.copy() for name, parameter in model.params.items()}
        for invalid_y in (np.array([-1, 1, 0]), np.array([0, 2, 0]),
                          np.array([0.0, 1.0, 0.0]), np.array([[0], [1], [0]])):
            with self.assertRaises(ValueError):
                model.loss(X, invalid_y)
        for options in ({"learning_rate": 0}, {"reg": -1}, {"num_iters": 0},
                        {"batch_size": 0}):
            with self.assertRaises(ValueError):
                model.train(X, y, **options)
        with self.assertRaises(ValueError):
            model.forward(np.ones((3, 4)))
        with self.assertRaises(ValueError):
            model.forward(np.ones((0, 2)))
        for name, parameter in model.params.items():
            np.testing.assert_array_equal(parameter, original_params[name])


if __name__ == "__main__":
    unittest.main()
