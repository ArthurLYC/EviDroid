import unittest

from evidroid.dynamic_weights import behavior_consistency_score, learn_view_weight_spec, weights_for_behavior
from evidroid.features import build_evidroid_feature_dict


class DynamicWeightTests(unittest.TestCase):
    def setUp(self):
        self.behavior_docs = [
            {
                "sample_id": "m1",
                "behaviors": [
                    {
                        "label": "dynamic_code_loading",
                        "evidence_ids": ["API_0001", "STR_0001"],
                        "support_by_view": {"api": 1, "string": 1},
                        "consistency_score": 0.6,
                    }
                ],
            },
            {
                "sample_id": "m2",
                "behaviors": [
                    {
                        "label": "dynamic_code_loading",
                        "evidence_ids": ["API_0002"],
                        "support_by_view": {"api": 1},
                        "consistency_score": 0.5,
                    }
                ],
            },
            {
                "sample_id": "b1",
                "behaviors": [
                    {
                        "label": "network_communication",
                        "evidence_ids": ["PERM_0001"],
                        "support_by_view": {"permission": 1},
                        "consistency_score": 0.3,
                    }
                ],
            },
            {
                "sample_id": "b2",
                "behaviors": [
                    {
                        "label": "network_communication",
                        "evidence_ids": ["PERM_0002"],
                        "support_by_view": {"permission": 1},
                        "consistency_score": 0.3,
                    }
                ],
            },
        ]
        self.labels = [1, 1, 0, 0]

    def test_global_weights_are_normalized(self):
        spec = learn_view_weight_spec(self.behavior_docs, self.labels, mode="global", alpha=0.5)
        self.assertAlmostEqual(sum(spec["global_weights"].values()), 1.0)
        self.assertEqual(set(spec["global_weights"]), {"permission", "api", "component", "string"})

    def test_behavior_specific_weights_can_be_learned(self):
        spec = learn_view_weight_spec(
            self.behavior_docs,
            self.labels,
            mode="behavior",
            alpha=1.0,
            min_label_samples=1,
        )
        weights = weights_for_behavior(spec, "dynamic_code_loading")
        self.assertGreater(weights["api"], weights["component"])
        self.assertIn("dynamic_code_loading", spec["behavior_weights"])

    def test_dynamic_consistency_score_uses_learned_weights(self):
        spec = {
            "mode": "behavior",
            "global_weights": {"permission": 0.25, "api": 0.25, "component": 0.25, "string": 0.25},
            "behavior_weights": {
                "dynamic_code_loading": {
                    "permission": 0.0,
                    "api": 0.8,
                    "component": 0.0,
                    "string": 0.2,
                }
            },
        }
        behavior = self.behavior_docs[0]["behaviors"][0]
        score = behavior_consistency_score(behavior, spec)
        self.assertGreater(score, behavior["consistency_score"])

    def test_evidroid_features_accept_dynamic_weight_spec(self):
        evidence_doc = {
            "sample_id": "m1",
            "evidence": [
                {"id": "API_0001", "view": "api", "value": "Landroid/Foo;->bar()V", "detail": {}},
                {"id": "STR_0001", "view": "string", "value": "classes.dex", "detail": {}},
            ],
        }
        spec = {
            "mode": "global",
            "global_weights": {"permission": 0.0, "api": 0.7, "component": 0.0, "string": 0.3},
            "behavior_weights": {},
        }
        features = build_evidroid_feature_dict(
            evidence_doc,
            self.behavior_docs[0],
            view_weight_spec=spec,
        )
        self.assertIn("ablation::consistency::dynamic_code_loading", features)
        self.assertGreater(features["ablation::consistency::dynamic_code_loading"], 0.0)


if __name__ == "__main__":
    unittest.main()
