import unittest

from scripts.run_multiseed_experiments import _select_threshold


class ThresholdSelectionTests(unittest.TestCase):
    def test_recall_at_precision_selects_lower_valid_threshold(self):
        y_true = [1, 1, 1, 0, 0]
        scores = [0.9, 0.6, 0.35, 0.5, 0.2]

        selected = _select_threshold(
            y_true,
            scores,
            mode="train_recall_at_precision",
            min_precision=0.75,
        )

        self.assertTrue(selected["constraint_satisfied"])
        self.assertAlmostEqual(selected["threshold"], 0.35)
        self.assertAlmostEqual(selected["train_precision"], 0.75)
        self.assertAlmostEqual(selected["train_recall"], 1.0)

    def test_unsatisfied_constraint_falls_back_to_train_f1(self):
        y_true = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.7, 0.6]

        selected = _select_threshold(
            y_true,
            scores,
            mode="train_recall_at_precision",
            min_precision=1.1,
        )

        self.assertFalse(selected["constraint_satisfied"])
        self.assertEqual(selected["requested_mode"], "train_recall_at_precision")
        self.assertEqual(selected["mode"], "train_f1")
        self.assertAlmostEqual(selected["train_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
