import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lc import LinearClassifier, softmax_loss, svm_loss, train_full_cifar10
from utils import load_cifar10


class LinearClassifierTests(unittest.TestCase):
    def test_epochs_visit_every_sample_including_the_last_partial_batch(self):
        for num_samples, batch_size in ((11, 4), (3, 8)):
            with self.subTest(num_samples=num_samples, batch_size=batch_size):
                batches = []

                def record_loss(W, b, X, y, reg):
                    batches.append(X[:, 0].copy())
                    return 0.0, np.zeros_like(W), np.zeros_like(b)

                X = np.arange(num_samples, dtype=np.float32)[:, None]
                y = np.zeros(num_samples, dtype=int)
                model = LinearClassifier(1, 2, record_loss)
                with contextlib.redirect_stdout(io.StringIO()):
                    model.train(X, y, num_epochs=2, batch_size=batch_size)

                batches_per_epoch = (num_samples + batch_size - 1) // batch_size
                self.assertEqual(len(batches), 2 * batches_per_epoch)
                for epoch in range(2):
                    start = epoch * batches_per_epoch
                    seen = np.concatenate(batches[start:start + batches_per_epoch])
                    np.testing.assert_array_equal(np.sort(seen), X[:, 0])

    def test_iteration_based_training_still_respects_num_iters(self):
        with patch("lc.softmax_loss", wraps=softmax_loss) as loss_fn:
            model = LinearClassifier(2, 2, loss_fn)
            with contextlib.redirect_stdout(io.StringIO()):
                model.train(np.ones((5, 2)), np.zeros(5, dtype=int),
                            num_iters=3, batch_size=2)
            self.assertEqual(loss_fn.call_count, 3)

    def test_full_training_returns_both_models_and_test_predictions(self):
        X = np.array([[-3, -1], [-2, -1], [-1, -1],
                      [1, 1], [2, 1], [3, 1]], dtype=np.float32)
        y = np.array([0, 0, 0, 1, 1, 1])
        X_test = np.array([[-2, -1], [2, 1]], dtype=np.float32)
        y_test = np.array([0, 1])
        data = (X, y, X[:0], y[:0], X_test, y_test)
        with patch("lc.load_cifar10", return_value=data) as loader:
            with contextlib.redirect_stdout(io.StringIO()):
                results = train_full_cifar10(num_epochs=3, batch_size=4,
                                            learning_rate=0.1)
        loader.assert_called_once_with(num_val=0)
        self.assertEqual(set(results), {"svm", "softmax"})
        self.assertIsNot(results["svm"]["model"], results["softmax"]["model"])
        for name, loss_fn in (("svm", svm_loss), ("softmax", softmax_loss)):
            result = results[name]
            self.assertIs(result["model"].loss_fn, loss_fn)
            np.testing.assert_array_equal(result["predictions"], y_test)
            self.assertEqual(result["test_accuracy"], 1.0)
            self.assertEqual(result["train_accuracy"], 1.0)


class CifarLoaderTests(unittest.TestCase):
    def test_full_and_validation_splits_center_using_only_training_images(self):
        pixels = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        train = SimpleNamespace(data=pixels.reshape(5, 1, 1, 1),
                                targets=[0, 1, 2, 3, 4])
        test = SimpleNamespace(data=np.array([255], dtype=np.uint8).reshape(1, 1, 1, 1),
                               targets=[1])
        normalized = pixels.astype(np.float32)[:, None] / 255.0
        for num_val in (0, 2):
            with self.subTest(num_val=num_val):
                with patch("utils.CIFAR10", side_effect=[train, test]):
                    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(num_val)
                num_train = len(pixels) - num_val
                mean = normalized[:num_train].mean(axis=0, keepdims=True)
                np.testing.assert_allclose(X_train, normalized[:num_train] - mean)
                np.testing.assert_allclose(X_val, normalized[num_train:] - mean)
                np.testing.assert_allclose(X_test, 1.0 - mean)
                np.testing.assert_array_equal(y_train, train.targets[:num_train])
                np.testing.assert_array_equal(y_val, train.targets[num_train:])
                np.testing.assert_array_equal(y_test, test.targets)


if __name__ == "__main__":
    unittest.main()
