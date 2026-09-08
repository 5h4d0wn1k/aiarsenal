import unittest

from aiarsenal import mcp


class TestMcp(unittest.TestCase):
    def test_mock_server_exposes_tools(self):
        res = mcp.run_mcp({})
        self.assertEqual(res["tools_exposed"], 5)

    def test_dangerous_tools_callable(self):
        res = mcp.run_mcp({})
        self.assertGreaterEqual(res["dangerous_tools_exposed"], 3)
        self.assertTrue(res["abuse_capable"])
        self.assertIn("shell_exec", res["exposed_tools"])


if __name__ == "__main__":
    unittest.main()