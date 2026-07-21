import tempfile
import unittest
from pathlib import Path

from evidroid.settings import load_llm_config


class SettingsTests(unittest.TestCase):
    def test_load_llm_config_from_nested_llm_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deepseek.json"
            path.write_text(
                '{"llm": {"api_key": "sk-test", "model": "deepseek-v4-pro"}}',
                encoding="utf-8",
            )
            config = load_llm_config(path)
        self.assertEqual(config["api_key"], "sk-test")
        self.assertEqual(config["model"], "deepseek-v4-pro")
        self.assertEqual(config["base_url"], "https://api.deepseek.com")


if __name__ == "__main__":
    unittest.main()
