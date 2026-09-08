import unittest

from aiarsenal.metrics import accuracy, binary_auc, roc_points


class TestMetrics(unittest.TestCase):
    def test_accuracy_basic(self):
        self.assertEqual(accuracy([1, 0, 1], [1, 0, 1]), 1.0)
        self.assertEqual(accuracy([1, 0, 1], [1, 1, 1]), 2 / 3)

    def test_accuracy_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            accuracy([1, 0], [1])

    def test_auc_perfect_separation(self):
        scores = [0.9, 0.8, 0.2, 0.1]
        labels = [1, 1, 0, 0]
        self.assertGreater(binary_auc(scores, labels), 0.95)

    def test_auc_random_is_half(self):
        # Interleaved scores/labels with no ordering signal -> 0.5 area.
        self.assertAlmostEqual(binary_auc([0.7, 0.6, 0.6, 0.4], [1, 0, 0, 1]),
                               0.5, places=6)

    def test_auc_partial_separation(self):
        # pos [0.9, 0.7], neg [0.8, 0.2]: outranking pairs = 3/4.
        self.assertAlmostEqual(binary_auc([0.9, 0.8, 0.7, 0.2], [1, 0, 1, 0]),
                               0.75, places=6)

    def test_roc_points_monotonic(self):
        pts = roc_points([0.9, 0.5, 0.1], [1, 0, 0])
        self.assertEqual(pts[-1], (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()