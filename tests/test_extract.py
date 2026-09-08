import unittest

from aiarsenal import extract


class TestExtract(unittest.TestCase):
    def test_student_fidelity_is_high(self):
        res = extract.run_extract({"n_queries": 500})
        self.assertGreater(res["student_fidelity_to_teacher"], 0.9)

    def test_model_flag_stolen(self):
        res = extract.run_extract({"n_queries": 600})
        self.assertTrue(res["model_stolen"])


if __name__ == "__main__":
    unittest.main()