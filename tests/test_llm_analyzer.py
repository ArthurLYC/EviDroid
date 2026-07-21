import json
import unittest

from evidroid.analyzers.llm_analyzer import OpenAIBehaviorAnalyzer
from evidroid.config import BEHAVIOR_RULES


class LLMAnalyzerPromptTests(unittest.TestCase):
    def test_prompt_taxonomy_matches_allowed_behavior_families(self):
        evidence_doc = {
            "sample_id": "demo",
            "label": "malware",
            "evidence": [
                {"id": "PERM_0001", "view": "permission", "value": "android.permission.INTERNET", "detail": {}},
                {
                    "id": "API_0001",
                    "view": "api",
                    "value": "Ljava/lang/Runtime;->exec(Ljava/lang/String;)Ljava/lang/Process;",
                    "detail": {},
                },
            ],
        }
        allowed = {item["label"] for item in BEHAVIOR_RULES}
        disallowed = {"privilege_escalation", "device_admin_abuse", "credential_theft"}

        for prompt_mode in ("default", "malware_focused", "risk_focused"):
            analyzer = OpenAIBehaviorAnalyzer(prompt_mode=prompt_mode)
            prompt = analyzer._build_prompt(evidence_doc)
            payload = json.loads(prompt)
            taxonomy = {item["label"] for item in payload["allowed_taxonomy"]}

            self.assertEqual(taxonomy, allowed)
            for label in disallowed:
                self.assertNotIn(label, prompt)

    def test_compact_budget_reduces_prompt_and_preserves_ids(self):
        evidence_doc = {
            "sample_id": "demo",
            "label": "malware",
            "evidence": [
                {"id": "PERM_0001", "view": "permission", "value": "android.permission.INTERNET", "detail": {}},
                {
                    "id": "API_0001",
                    "view": "api",
                    "value": "Ljava/lang/Runtime;->exec(Ljava/lang/String;)Ljava/lang/Process;",
                    "detail": {},
                },
                {
                    "id": "API_0002",
                    "view": "api",
                    "value": "Ljava/lang/String;->length()I",
                    "detail": {},
                },
                {
                    "id": "STR_0001",
                    "view": "string",
                    "value": "https://example.com/" + ("very-long-path/" * 20) + "?token=secret",
                    "detail": {},
                },
            ]
            + [
                {
                    "id": f"STR_{idx:04d}",
                    "view": "string",
                    "value": "generic-long-resource-" + ("token/" * 30) + str(idx),
                    "detail": {},
                }
                for idx in range(2, 60)
            ],
        }
        legacy = OpenAIBehaviorAnalyzer(prompt_mode="risk_focused")
        compact = OpenAIBehaviorAnalyzer(
            prompt_mode="risk_focused",
            evidence_budget_mode="adaptive",
            view_budgets={"permission": 1, "api": 1, "string": 1},
            compact_evidence=True,
            max_value_chars=80,
        )

        legacy_payload = json.loads(legacy._build_prompt(evidence_doc))
        compact_prompt = compact._build_prompt(evidence_doc)
        compact_payload = json.loads(compact_prompt)

        self.assertLess(len(compact_prompt), len(json.dumps(legacy_payload, ensure_ascii=False)))
        self.assertEqual(compact_payload["evidence"]["api"][0][0], "API_0001")
        self.assertEqual(compact_payload["evidence"]["string"][0][0], "STR_0001")
        self.assertIn("evidence_budget", compact_payload)


if __name__ == "__main__":
    unittest.main()
