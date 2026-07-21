import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_evidence_quality_and_failures.py"
SPEC = importlib.util.spec_from_file_location("evidence_quality_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analysis)


def test_evidence_quality_counts_valid_references(tmp_path):
    evidence = tmp_path / "evidence.jsonl"
    behaviors = {
        "s1": {
            "sample_id": "s1",
            "label": "benign",
            "behaviors": [
                {
                    "label": "network_communication",
                    "evidence_ids": ["PERM_0001", "API_0001"],
                    "support_by_view": {"permission": 1, "api": 1},
                    "consistency_score": 0.75,
                }
            ],
        }
    }
    evidence.write_text(
        '{"sample_id":"s1","label":"benign","evidence":['
        '{"id":"PERM_0001","view":"permission"},'
        '{"id":"API_0001","view":"api"}]}\n',
        encoding="utf-8",
    )

    result = analysis.analyze_evidence_quality(evidence, behaviors)
    summary = result["summary"]

    assert summary["behavior_records"] == 1
    assert summary["valid_evidence_reference_rate"] == 1.0
    assert summary["support_match_record_rate"] == 1.0
    assert summary["multi_view_behavior_rate"] == 1.0


def test_failure_case_categories_use_prediction_rows():
    predictions = [
        {"sample_id": "s1", "y_true": 0, "y_pred": 1, "score": 0.52},
        {"sample_id": "s2", "y_true": 1, "y_pred": 1, "score": 0.82},
    ]
    behaviors = {
        "s1": {
            "sample_id": "s1",
            "behaviors": [
                {
                    "label": "network_communication",
                    "evidence_ids": ["PERM_0001"],
                    "support_by_view": {"permission": 1},
                    "consistency_score": 0.35,
                }
            ],
        }
    }

    result = analysis.analyze_failure_cases(predictions, behaviors, max_examples=5)

    assert result["summary"]["failed_rows"] == 1
    assert result["summary"]["false_positives"] == 1
    assert result["examples"][0]["error_type"] == "false_positive"
    assert "threshold_margin" in result["examples"][0]["categories"]
