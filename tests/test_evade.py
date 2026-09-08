import unittest

from aiarsenal import evade


class TestEvade(unittest.TestCase):
    def test_clean_accuracy_is_high(self):
        res = evade.run_evade({})
        self.assertGreater(res["clean_accuracy"], 0.85)

    def test_robust_accuracy_declines_with_epsilon(self):
        res = evade.run_evade({"epsilons": [0.0, 0.5, 1.0]})
        self.assertGreater(res["robust_accuracy_by_epsilon"]["eps_0.0"],
                           res["robust_accuracy_by_epsilon"]["eps_0.5"])
        self.assertGreater(res["robust_accuracy_by_epsilon"]["eps_0.5"],
                           res["robust_accuracy_by_epsilon"]["eps_1.0"])


if __name__ == "__main__":
    unittest.main()