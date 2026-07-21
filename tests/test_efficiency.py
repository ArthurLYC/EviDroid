import unittest

from evidroid.efficiency import summarize_evidence_scale, summarize_numbers


class EfficiencyTests(unittest.TestCase):
    def test_summarize_numbers(self):
        stats = summarize_numbers([1, 2, 3, 4])
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["mean"], 2.5)
        self.assertEqual(stats["median"], 2.5)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 4.0)

    def test_summarize_evidence_scale(self):
        stats = summarize_evidence_scale(
            [
                {"view_counts": {"permission": 1, "api": 2, "component": 1, "string": 3}, "errors": []},
                {"view_counts": {"permission": 3, "api": 4, "component": 1, "string": 5}, "errors": ["x"]},
            ]
        )
        self.assertEqual(stats["rows_with_errors"], 1)
        self.assertEqual(stats["by_view"]["permission"]["mean"], 2.0)
        self.assertEqual(stats["total_evidence"]["max"], 13.0)


if __name__ == "__main__":
    unittest.main()
