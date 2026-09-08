import os
import tempfile
import unittest

from aiarsenal import campaign


class TestCampaign(unittest.TestCase):
    def test_refuses_unapproved_scope(self):
        with self.assertRaises(campaign.CampaignRefused) as ctx:
            campaign.run_campaign({"scope": "evil.example.org",
                                   "modules": ["poison"], "approxed": True})
        self.assertEqual(ctx.exception.exit_code, 2)

    def test_requires_approval_gate(self):
        with self.assertRaises(campaign.CampaignRefused):
            campaign.run_campaign({"scope": "lab:test", "modules": ["poison"],
                                   "approved": False})

    def test_dry_run_plans_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = os.path.join(tmp, "audit.jsonl")
            res = campaign.run_campaign({"scope": "lab:test-a",
                                         "modules": ["poison"],
                                         "dry_run": True, "approved": True,
                                         "audit_path": audit})
            self.assertTrue(res["dry_run"])
            with open(audit, "r", encoding="utf-8") as fh:
                lines = [l for l in fh if l.strip()]
            self.assertEqual(len(lines), 2)  # plan + dry-run step

    def test_unknown_module_refused(self):
        with self.assertRaises(campaign.CampaignRefused):
            campaign.run_campaign({"scope": "lab:test",
                                   "modules": ["no_such_mod"],
                                   "dry_run": True, "approved": True})


if __name__ == "__main__":
    unittest.main()