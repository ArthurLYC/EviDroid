import unittest

from evidroid.reporting import build_report_context, unknown_evidence_ids


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.evidence_doc = {
            "sample_id": "demo",
            "package": "com.example.demo",
            "sha256": "abc",
            "view_counts": {"permission": 1, "api": 1},
            "evidence": [
                {"id": "PERM_0001", "view": "permission", "value": "android.permission.INTERNET", "detail": {}},
                {"id": "API_0001", "view": "api", "value": "Landroid/net/Uri;->parse(Ljava/lang/String;)Landroid/net/Uri;", "detail": {}},
            ],
        }
        self.behavior_doc = {
            "sample_id": "demo",
            "behaviors": [
                {
                    "label": "network_communication",
                    "name": "Network communication",
                    "description": "Uses network-related APIs and permissions.",
                    "evidence_ids": ["PERM_0001", "API_0001"],
                    "support_by_view": {"permission": 1, "api": 1},
                    "consistency_score": 0.75,
                }
            ],
        }

    def test_report_context_uses_behavior_evidence(self):
        context = build_report_context(self.evidence_doc, self.behavior_doc)
        self.assertEqual(context["sample"]["sample_id"], "demo")
        behavior = context["behavior_findings"][0]
        self.assertEqual(behavior["evidence"][0]["id"], "PERM_0001")
        self.assertTrue(context["report_constraints"]["static_analysis_only"])

    def test_unknown_evidence_ids_are_detected(self):
        unknown = unknown_evidence_ids("Known PERM_0001 and unknown API_9999", self.evidence_doc)
        self.assertEqual(unknown, ["API_9999"])

    def test_malware_context_includes_rationale(self):
        prediction_doc = {
            "prediction_label": "malware",
            "malware_score": 0.91,
            "model_path": "model.joblib",
        }
        context = build_report_context(self.evidence_doc, self.behavior_doc, prediction_doc=prediction_doc)
        self.assertEqual(context["report_profile"]["template"], "malware_triage")
        self.assertEqual(context["malware_rationale"][0]["label"], "network_communication")
        self.assertIn("可支撑远程控制", context["malware_rationale"][0]["why_it_matters"])
        self.assertIn("PERM_0001", context["malware_rationale"][0]["evidence_ids"])


if __name__ == "__main__":
    unittest.main()
