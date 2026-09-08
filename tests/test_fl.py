import unittest

from aiarsenal import fl


class TestFL(unittest.TestCase):
    def test_backdoor_survives_aggregation(self):
        res = fl.run_fl({"boost": 10.0, "epochs": 300})
        self.assertGreater(res["backdoor_activation_victims"], 0.95)

    def test_clean_utility_preserved(self):
        res = fl.run_fl({"boost": 10.0, "epochs": 300})
        self.assertGreater(res["clean_accuracy_after_attack"], 0.5)

    def test_benign_control_does_not_fire(self):
        res = fl.run_fl({"malicious_client": None, "epochs": 300})
        self.assertFalse(res["attacker_present"])
        self.assertLess(res["backdoor_activation_victims"], 0.3)


if __name__ == "__main__":
    unittest.main()