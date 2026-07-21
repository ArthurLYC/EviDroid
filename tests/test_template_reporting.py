import tempfile
import unittest
from pathlib import Path

from evidroid.template_reporting import build_template_report_markdown, generate_template_report


class TemplateReportingTests(unittest.TestCase):
    def setUp(self):
        self.evidence_doc = {
            "sample_id": "demo",
            "package": "com.example.demo",
            "sha256": "abc",
            "label": "malware",
            "view_counts": {"permission": 1, "api": 1, "string": 1},
            "evidence": [
                {"id": "PERM_0001", "view": "permission", "value": "android.permission.INTERNET", "detail": {}},
                {
                    "id": "API_0001",
                    "view": "api",
                    "value": "Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;",
                    "detail": {},
                },
                {"id": "STR_0001", "view": "string", "value": "https://example.test/upload", "detail": {}},
            ],
        }
        self.behavior_doc = {
            "sample_id": "demo",
            "behaviors": [
                {
                    "label": "network_communication",
                    "name": "Network communication",
                    "description": "Network behavior",
                    "evidence_ids": ["PERM_0001", "STR_0001"],
                    "support_by_view": {"permission": 1, "string": 1},
                    "consistency_score": 0.7,
                },
                {
                    "label": "privacy_collection",
                    "name": "Privacy collection",
                    "description": "Privacy behavior",
                    "evidence_ids": ["API_0001"],
                    "support_by_view": {"api": 1},
                    "consistency_score": 0.5,
                },
            ],
        }
        self.prediction_doc = {
            "prediction_label": "malware",
            "malware_score": 0.91,
        }

    def test_template_markdown_contains_prediction_behavior_and_evidence(self):
        markdown = build_template_report_markdown(
            self.evidence_doc,
            self.behavior_doc,
            prediction_doc=self.prediction_doc,
        )
        self.assertIn("Detection result: `malware`", markdown)
        self.assertIn("Network communication", markdown)
        self.assertIn("PERM_0001", markdown)
        self.assertNotIn("Ground-truth label", markdown)
        self.assertIn("本报告基于静态证据、分类器输出和已验证行为记录生成", markdown)

    def test_generate_template_report_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.md"
            result = generate_template_report(
                self.evidence_doc,
                self.behavior_doc,
                out,
                prediction_doc=self.prediction_doc,
            )
            self.assertTrue(out.exists())
            self.assertEqual(result["sample_id"], "demo")
            self.assertIn("EviDroid Template Report", out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
