import unittest

from aiarsenal import invert


class TestInvert(unittest.TestCase):
    def test_membership_auc_above_chance(self):
        res = invert.run_invert({})
        self.assertGreater(res["membership_auc"], 0.7)

    def test_membership_auc_not_trivially_one(self):
        res = invert.run_invert({"jitter": 0.14})
        self.assertLess(res["membership_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()