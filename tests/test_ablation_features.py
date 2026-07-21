import unittest

from evidroid.features import build_ablation_feature_dict, build_ablation_feature_parts


class AblationFeatureTests(unittest.TestCase):
    def setUp(self):
        self.evidence_doc = {
            "sample_id": "demo",
            "evidence": [
                {
                    "id": "PERM_0001",
                    "view": "permission",
                    "value": "android.permission.INTERNET",
                    "detail": {},
                },
                {
                    "id": "API_0001",
                    "view": "api",
                    "value": "Landroid/telephony/SmsManager;->sendTextMessage(Ljava/lang/String;)V",
                    "detail": {},
                },
                {
                    "id": "STR_0001",
                    "view": "string",
                    "value": "https://example.com/payload.dex",
                    "detail": {},
                },
            ],
        }
        self.behavior_doc = {
            "sample_id": "demo",
            "behaviors": [
                {
                    "label": "network_communication",
                    "consistency_score": 0.75,
                    "evidence_ids": ["PERM_0001", "API_0001"],
                    "support_by_view": {"permission": 1, "api": 1},
                }
            ],
        }

    def test_a1_has_behavior_but_no_consistency(self):
        features = build_ablation_feature_dict(
            self.evidence_doc,
            self.behavior_doc,
            use_behavior_semantics=True,
            use_consistency=False,
        )
        self.assertIn("ablation::behavior::network_communication", features)
        self.assertNotIn("ablation::consistency::network_communication", features)
        self.assertNotIn("ablation::consistency_mean", features)

    def test_a2_has_label_free_consistency(self):
        features = build_ablation_feature_dict(
            self.evidence_doc,
            self.behavior_doc,
            use_behavior_semantics=False,
            use_consistency=True,
        )
        self.assertNotIn("ablation::behavior::network_communication", features)
        self.assertNotIn("ablation::consistency::network_communication", features)
        self.assertEqual(features["ablation::consistency_mean"], 0.75)
        self.assertEqual(features["ablation::view_count_max"], 2.0)

    def test_a3_has_behavior_and_label_specific_consistency(self):
        features = build_ablation_feature_dict(
            self.evidence_doc,
            self.behavior_doc,
            use_behavior_semantics=True,
            use_consistency=True,
        )
        self.assertIn("ablation::behavior::network_communication", features)
        self.assertEqual(features["ablation::consistency::network_communication"], 0.75)
        self.assertEqual(features["ablation::support::network_communication::api"], 1.0)

    def test_dynamic_view_weights_replace_consistency_score(self):
        weight_spec = {
            "mode": "behavior",
            "global_weights": {
                "permission": 0.1,
                "api": 0.2,
                "component": 0.3,
                "string": 0.4,
            },
            "behavior_weights": {},
        }
        features = build_ablation_feature_dict(
            self.evidence_doc,
            self.behavior_doc,
            use_behavior_semantics=True,
            use_consistency=True,
            view_weight_spec=weight_spec,
        )
        self.assertAlmostEqual(features["ablation::consistency::network_communication"], 0.31)
        self.assertAlmostEqual(features["ablation::consistency_mean"], 0.31)

    def test_behavior_filter_removes_low_consistency_findings(self):
        features = build_ablation_feature_dict(
            self.evidence_doc,
            self.behavior_doc,
            use_behavior_semantics=True,
            use_consistency=True,
            min_consistency=0.8,
        )
        self.assertNotIn("ablation::behavior::network_communication", features)
        self.assertEqual(features["ablation::consistency_count"], 0.0)

    def test_compact_static_profile_groups_high_cardinality_features(self):
        features = build_ablation_feature_dict(
            self.evidence_doc,
            self.behavior_doc,
            use_behavior_semantics=False,
            use_consistency=False,
            static_profile="compact",
        )
        self.assertIn("static::permission_group::network", features)
        self.assertIn("static::api_family::sms_call", features)
        self.assertIn("static::string_marker::url_http", features)
        self.assertEqual(features["count::suspicious_string"], 1.0)

    def test_v2_features_add_behavior_and_consistency_detail(self):
        features = build_ablation_feature_dict(
            self.evidence_doc,
            self.behavior_doc,
            use_behavior_semantics=True,
            use_consistency=True,
            static_profile="compact",
            feature_version="v2",
        )
        self.assertIn("behavior_v2::network_communication::present", features)
        self.assertIn("behavior_v2::network_communication::api_family::sms_call", features)
        self.assertIn("consistency_v2::network_communication::view_mask::permission+api", features)

    def test_v2_features_add_llm_prompt_agreement_detail(self):
        behavior_doc = {
            "sample_id": "demo",
            "behaviors": [
                {
                    "label": "network_communication",
                    "consistency_score": 0.75,
                    "evidence_ids": ["PERM_0001"],
                    "support_by_view": {"permission": 1},
                    "llm_prompt": "default",
                },
                {
                    "label": "network_communication",
                    "consistency_score": 0.65,
                    "evidence_ids": ["API_0001"],
                    "support_by_view": {"api": 1},
                    "llm_prompt": "malware_focused",
                },
            ],
        }
        features = build_ablation_feature_dict(
            self.evidence_doc,
            behavior_doc,
            use_behavior_semantics=True,
            use_consistency=True,
            static_profile="compact",
            feature_version="v2",
        )
        self.assertIn("behavior_v2::network_communication::source::default", features)
        self.assertIn("behavior_v2::network_communication::source::malware_focused", features)
        self.assertEqual(features["behavior_v2::network_communication::source_count"], 2.0)
        self.assertIn("behavior_v2::network_communication::llm_prompt_agreement", features)
        self.assertIn("consistency_v2::network_communication::llm_prompt_agreement", features)

    def test_v2_features_add_llm_risk_detail(self):
        behavior_doc = {
            "sample_id": "demo",
            "llm_risk": {
                "apk_risk_score": 0.82,
                "risk_level": "high",
            },
            "behaviors": [
                {
                    "label": "network_communication",
                    "consistency_score": 0.75,
                    "evidence_ids": ["PERM_0001", "API_0001"],
                    "support_by_view": {"permission": 1, "api": 1},
                    "malware_relevance": 0.7,
                    "confidence": 0.9,
                    "risk_level": "high",
                }
            ],
        }
        features = build_ablation_feature_dict(
            self.evidence_doc,
            behavior_doc,
            use_behavior_semantics=True,
            use_consistency=True,
            static_profile="compact",
            feature_version="v2",
        )
        self.assertEqual(features["behavior_v2::llm_doc_risk_score"], 0.82)
        self.assertIn("behavior_v2::llm_doc_risk_level::high", features)
        self.assertEqual(features["behavior_v2::network_communication::malware_relevance_max"], 0.7)
        self.assertEqual(features["behavior_v2::network_communication::llm_confidence_max"], 0.9)
        self.assertIn("behavior_v2::network_communication::risk_level::high", features)

    def test_feature_parts_match_a3_builder(self):
        parts = build_ablation_feature_parts(
            self.evidence_doc,
            self.behavior_doc,
            static_profile="compact",
            feature_version="v2",
        )
        composed = {
            **parts["static"],
            **parts["behavior"],
            **parts["behavior_consistency"],
        }
        expected = build_ablation_feature_dict(
            self.evidence_doc,
            self.behavior_doc,
            use_behavior_semantics=True,
            use_consistency=True,
            static_profile="compact",
            feature_version="v2",
        )
        self.assertEqual(composed, expected)


if __name__ == "__main__":
    unittest.main()
