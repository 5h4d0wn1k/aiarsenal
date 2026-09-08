import unittest

from aiarsenal import poison


class TestPoison(unittest.TestCase):
    def test_poisoning_collapses_accuracy(self):
        res = poison.run_poison({"poison_frac": 0.5, "seed": 0})
        self.assertGreater(res["baseline_clean_accuracy"], 0.7)
        self.assertLess(res["poisoned_clean_accuracy"], res["baseline_clean_accuracy"])
        self.assertGreater(res["accuracy_collapse"], 0.25)

    def test_backdoor_trigger_retained(self):
        res = poison.run_poison({"poison_frac": 0.5, "seed": 0})
        self.assertGreater(res["backdoor_trigger_activation"], 0.95)


if __name__ == "__main__":
    unittest.main()