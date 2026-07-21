import unittest

from sklearn.pipeline import Pipeline

from evidroid.classifier_selection import available_classifiers, feature_selection_metadata, make_classifier_pipeline


class ClassifierSelectionTests(unittest.TestCase):
    def test_required_classifiers_are_available(self):
        availability = available_classifiers()
        for name in [
            "logistic_regression",
            "linear_svm",
            "knn",
            "decision_tree",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "adaboost",
            "mlp",
            "gaussian_naive_bayes",
            "naive_bayes",
        ]:
            self.assertTrue(availability[name])

    def test_make_logistic_regression_pipeline(self):
        pipeline = make_classifier_pipeline("logistic_regression", random_state=7)
        self.assertIsInstance(pipeline, Pipeline)
        self.assertIn("vectorizer", pipeline.named_steps)
        self.assertIn("classifier", pipeline.named_steps)

    def test_make_random_forest_pipeline_can_select_features(self):
        pipeline = make_classifier_pipeline("random_forest", random_state=7, select_k_best=10)
        self.assertIsInstance(pipeline, Pipeline)
        self.assertIn("select", pipeline.named_steps)
        self.assertIn("classifier", pipeline.named_steps)

    def test_tree_fusion_candidate_pipelines_are_available(self):
        for name in [
            "decision_tree",
            "extra_trees",
            "extra_trees_regularized",
            "extra_trees_shallow",
            "extra_trees_deep",
            "extra_trees_calibrated",
            "gradient_boosting",
            "adaboost",
        ]:
            pipeline = make_classifier_pipeline(name, random_state=7)
            self.assertIsInstance(pipeline, Pipeline)
            self.assertIn("classifier", pipeline.named_steps)

    def test_make_mlp_pipeline_has_dense_step(self):
        pipeline = make_classifier_pipeline("mlp", random_state=7)
        self.assertIn("dense", pipeline.named_steps)

    def test_common_fusion_baselines_are_available(self):
        for name in [
            "logistic_regression",
            "linear_svm",
            "knn",
            "gaussian_naive_bayes",
            "mlp",
        ]:
            pipeline = make_classifier_pipeline(name, random_state=7)
            self.assertIsInstance(pipeline, Pipeline)
            self.assertIn("classifier", pipeline.named_steps)

    def test_optimized_fusion_heads_are_registered_when_xgboost_available(self):
        availability = available_classifiers()
        if not availability["xgboost"]:
            self.skipTest("xgboost is not installed")
        for name in [
            "xgboost_regularized",
            "xgboost_stump",
            "xgboost_shallow",
            "xgboost_depth2",
            "xgboost_compact",
            "xgboost_calibrated",
        ]:
            self.assertTrue(availability[name])
            pipeline = make_classifier_pipeline(name, random_state=7)
            self.assertIsInstance(pipeline, Pipeline)
            self.assertIn("classifier", pipeline.named_steps)

    def test_grouped_selector_tracks_per_group_feature_counts(self):
        rows = [
            {"static::a": 1.0, "static::b": 1.0, "behavior_v2::x::present": 1.0},
            {"static::a": 1.0, "consistency_v2::score_bucket::high": 1.0},
            {"static::c": 1.0, "behavior_v2::y::present": 1.0},
            {"static::c": 1.0, "consistency_v2::score_bucket::low": 1.0},
        ]
        y = [0, 0, 1, 1]
        pipeline = make_classifier_pipeline(
            "random_forest",
            random_state=7,
            grouped_select_k_best={"static": 1, "behavior": 2, "consistency": 2},
        )
        pipeline.fit(rows, y)
        metadata = feature_selection_metadata(pipeline)
        self.assertNotIn("select", pipeline.named_steps)
        self.assertEqual(metadata["group_selected_feature_counts"]["static"], 1)
        self.assertEqual(metadata["group_feature_counts"]["behavior"], 2)
        self.assertGreaterEqual(metadata["selected_feature_count"], 3)


if __name__ == "__main__":
    unittest.main()
