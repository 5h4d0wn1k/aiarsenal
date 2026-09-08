import unittest

from aiarsenal import backdoor


class TestBackdoor(unittest.TestCase):
    def test_trigger_activates_backdoor(self):
        res = backdoor.run_backdoor({})
        self.assertGreater(res["backdoor_trigger_activation"], 0.95)

    def test_clean_inputs_kept_working(self):
        res = backdoor.run_backdoor({})
        self.assertGreater(res["clean_accuracy_after_backdoor"], 0.9)
        self.assertTrue(res["clean_preserved"])


if __name__ == "__main__":
    unittest.main()