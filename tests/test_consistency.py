import unittest

from evidroid.consistency import consistency_score, support_by_view


class ConsistencyTests(unittest.TestCase):
    def test_consistency_uses_unique_views_and_support_count(self):
        index = {
            "PERM_0001": {"view": "permission"},
            "API_0001": {"view": "api"},
            "STR_0001": {"view": "string"},
        }
        score = consistency_score(["PERM_0001", "API_0001", "STR_0001"], index)
        self.assertGreater(score, 0.5)
        self.assertEqual(
            support_by_view(["PERM_0001", "API_0001", "STR_0001"], index),
            {"permission": 1, "api": 1, "string": 1},
        )


if __name__ == "__main__":
    unittest.main()
