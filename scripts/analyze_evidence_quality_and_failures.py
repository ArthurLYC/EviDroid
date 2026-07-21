from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VIEW_ORDER = ("permission", "api", "component", "string")
RISK_LABELS = {
    "network_communication",
    "file_storage_access",
    "crypto_or_obfuscation",
    "dynamic_code_loading",
    "sms_or_call_abuse",
    "privacy_identifier_access",
    "location_tracking",
    "credential_or_account_access",
    "reflection_or_native_code",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze EviDroid evidence-backed behavior quality and failure cases."
    )
    parser.add_argument("--evidence", default="data/processed/evidence_final_20000_balanced_20260706.jsonl")
    parser.add_argument("--behaviors", default="data/processed/behaviors_llm_final_20000_balanced_20260706.jsonl")
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional JSON/JSONL/CSV with per-sample prediction rows. Metric JSONs need --save-predictions.",
    )
    parser.add_argument("--out-dir", default="artifacts/analysis/evidence_quality_failures")
    parser.add_argument("--max-examples", type=int, default=12)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    behaviors_by_id = load_jsonl_by_sample_id(Path(args.behaviors))
    quality = analyze_evidence_quality(Path(args.evidence), behaviors_by_id)
    write_json(out_dir / "evidence_quality_summary.json", quality["summary"])
    write_csv(out_dir / "evidence_quality_by_behavior.csv", quality["by_behavior"])

    failure_result: dict[str, Any] | None = None
    if args.predictions:
        predictions = load_prediction_rows(Path(args.predictions))
        failure_result = analyze_failure_cases(predictions, behaviors_by_id, max_examples=args.max_examples)
        write_json(out_dir / "failure_case_summary.json", failure_result["summary"])
        write_csv(out_dir / "failure_case_categories.csv", failure_result["categories"])
        write_jsonl(out_dir / "failure_case_examples.jsonl", failure_result["examples"])

    write_report(out_dir / "analysis_report.md", quality, failure_result, bool(args.predictions))
    print(f"[analysis] wrote {out_dir}")


def analyze_evidence_quality(
    evidence_path: Path,
    behaviors_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary_counter: Counter[str] = Counter()
    behavior_stats: dict[str, dict[str, Any]] = defaultdict(new_behavior_stat)
    consistency_scores: list[float] = []
    evidence_ids_per_behavior: list[int] = []
    sample_behavior_counts: list[int] = []
    sample_multi_view_ratios: list[float] = []
    view_coverage_counter: Counter[int] = Counter()
    label_counter: Counter[str] = Counter()

    seen_evidence_samples: set[str] = set()
    for evidence_doc in iter_jsonl(evidence_path):
        sample_id = str(evidence_doc.get("sample_id", ""))
        seen_evidence_samples.add(sample_id)
        evidence_by_id = {
            str(item.get("id")): str(item.get("view"))
            for item in evidence_doc.get("evidence", [])
            if item.get("id")
        }
        behavior_doc = behaviors_by_id.get(sample_id)
        if not behavior_doc:
            summary_counter["samples_without_behavior_doc"] += 1
            continue
        label_counter[str(behavior_doc.get("label", evidence_doc.get("label", "unknown")))] += 1
        behaviors = list(behavior_doc.get("behaviors", []))
        summary_counter["samples_with_behavior_doc"] += 1
        sample_behavior_counts.append(len(behaviors))
        sample_multi_view_count = 0

        for behavior in behaviors:
            summary_counter["behavior_records"] += 1
            label = str(behavior.get("label", "unknown"))
            evidence_ids = [str(item) for item in behavior.get("evidence_ids", [])]
            support_by_view = normalize_support(behavior.get("support_by_view", {}))
            actual_views = Counter(evidence_by_id[item] for item in evidence_ids if item in evidence_by_id)
            unknown_ids = [item for item in evidence_ids if item not in evidence_by_id]
            view_count = len([view for view, count in actual_views.items() if count > 0])
            all_refs_valid = bool(evidence_ids) and not unknown_ids
            support_matches = all_refs_valid and all(
                actual_views.get(view, 0) == count for view, count in support_by_view.items()
            )
            support_matches = support_matches and all(
                support_by_view.get(view, 0) == count for view, count in actual_views.items()
            )

            summary_counter["evidence_refs"] += len(evidence_ids)
            summary_counter["valid_evidence_refs"] += len(evidence_ids) - len(unknown_ids)
            summary_counter["unknown_evidence_refs"] += len(unknown_ids)
            summary_counter["nonempty_behavior_records"] += int(bool(evidence_ids))
            summary_counter["all_refs_valid_records"] += int(all_refs_valid)
            summary_counter["support_match_records"] += int(support_matches)
            summary_counter["multi_view_behavior_records"] += int(view_count >= 2)
            summary_counter["single_view_behavior_records"] += int(view_count == 1)
            summary_counter["zero_view_behavior_records"] += int(view_count == 0)
            sample_multi_view_count += int(view_count >= 2)
            view_coverage_counter[view_count] += 1
            evidence_ids_per_behavior.append(len(evidence_ids))
            consistency_scores.append(float(behavior.get("consistency_score", 0.0) or 0.0))

            stat = behavior_stats[label]
            stat["behavior_label"] = label
            stat["records"] += 1
            stat["samples"].add(sample_id)
            stat["evidence_refs"] += len(evidence_ids)
            stat["valid_records"] += int(all_refs_valid)
            stat["support_match_records"] += int(support_matches)
            stat["multi_view_records"] += int(view_count >= 2)
            stat["consistency_scores"].append(float(behavior.get("consistency_score", 0.0) or 0.0))
            for view in VIEW_ORDER:
                stat[f"{view}_support"] += actual_views.get(view, 0)

        sample_multi_view_ratios.append(safe_div(sample_multi_view_count, len(behaviors)))

    missing_evidence_samples = set(behaviors_by_id) - seen_evidence_samples
    summary_counter["behavior_docs_without_evidence_doc"] = len(missing_evidence_samples)

    behavior_rows = []
    for label, stat in sorted(behavior_stats.items()):
        scores = stat["consistency_scores"]
        records = int(stat["records"])
        behavior_rows.append(
            {
                "behavior_label": label,
                "records": records,
                "sample_count": len(stat["samples"]),
                "mean_consistency": round_float(mean(scores)),
                "valid_record_rate": round_float(safe_div(stat["valid_records"], records)),
                "support_match_rate": round_float(safe_div(stat["support_match_records"], records)),
                "multi_view_rate": round_float(safe_div(stat["multi_view_records"], records)),
                "mean_evidence_refs": round_float(safe_div(stat["evidence_refs"], records)),
                "permission_refs": int(stat["permission_support"]),
                "api_refs": int(stat["api_support"]),
                "component_refs": int(stat["component_support"]),
                "string_refs": int(stat["string_support"]),
            }
        )

    total_records = int(summary_counter["behavior_records"])
    summary = {
        "evidence_samples": len(seen_evidence_samples),
        "behavior_samples": len(behaviors_by_id),
        "samples_with_behavior_doc": int(summary_counter["samples_with_behavior_doc"]),
        "samples_without_behavior_doc": int(summary_counter["samples_without_behavior_doc"]),
        "behavior_docs_without_evidence_doc": int(summary_counter["behavior_docs_without_evidence_doc"]),
        "label_counts": dict(label_counter),
        "behavior_records": total_records,
        "nonempty_behavior_record_rate": round_float(
            safe_div(summary_counter["nonempty_behavior_records"], total_records)
        ),
        "valid_evidence_reference_rate": round_float(
            safe_div(summary_counter["valid_evidence_refs"], summary_counter["evidence_refs"])
        ),
        "all_refs_valid_record_rate": round_float(
            safe_div(summary_counter["all_refs_valid_records"], total_records)
        ),
        "support_match_record_rate": round_float(
            safe_div(summary_counter["support_match_records"], total_records)
        ),
        "multi_view_behavior_rate": round_float(
            safe_div(summary_counter["multi_view_behavior_records"], total_records)
        ),
        "single_view_behavior_rate": round_float(
            safe_div(summary_counter["single_view_behavior_records"], total_records)
        ),
        "view_coverage_distribution": {str(key): value for key, value in sorted(view_coverage_counter.items())},
        "mean_behaviors_per_sample": round_float(mean(sample_behavior_counts)),
        "median_behaviors_per_sample": round_float(median(sample_behavior_counts)),
        "mean_evidence_refs_per_behavior": round_float(mean(evidence_ids_per_behavior)),
        "median_evidence_refs_per_behavior": round_float(median(evidence_ids_per_behavior)),
        "mean_consistency_score": round_float(mean(consistency_scores)),
        "median_consistency_score": round_float(median(consistency_scores)),
        "mean_sample_multi_view_ratio": round_float(mean(sample_multi_view_ratios)),
        "unknown_evidence_refs": int(summary_counter["unknown_evidence_refs"]),
    }
    return {"summary": summary, "by_behavior": behavior_rows}


def analyze_failure_cases(
    prediction_rows: list[dict[str, Any]],
    behaviors_by_id: dict[str, dict[str, Any]],
    max_examples: int,
) -> dict[str, Any]:
    summary_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for row in prediction_rows:
        y_true = label_to_int(row.get("y_true", row.get("true_label")))
        y_pred = label_to_int(row.get("y_pred", row.get("predicted_label")))
        if y_true is None or y_pred is None:
            continue
        summary_counter["prediction_rows"] += 1
        if y_true == y_pred:
            summary_counter["correct_rows"] += 1
            continue
        sample_id = str(row.get("sample_id", ""))
        error_type = "false_positive" if y_true == 0 and y_pred == 1 else "false_negative"
        summary_counter[error_type] += 1
        profile = behavior_profile(behaviors_by_id.get(sample_id, {}))
        categories = classify_failure(error_type, profile, row)
        for category in categories:
            category_counter[f"{error_type}:{category}"] += 1
        if len(examples) < max_examples:
            examples.append(
                {
                    "sample_id": sample_id,
                    "error_type": error_type,
                    "score": optional_float(row.get("score")),
                    "categories": categories,
                    "behavior_count": profile["behavior_count"],
                    "mean_consistency": profile["mean_consistency"],
                    "multi_view_ratio": profile["multi_view_ratio"],
                    "top_behaviors": profile["top_behaviors"],
                }
            )

    failed = summary_counter["false_positive"] + summary_counter["false_negative"]
    summary = {
        "prediction_rows": int(summary_counter["prediction_rows"]),
        "correct_rows": int(summary_counter["correct_rows"]),
        "failed_rows": int(failed),
        "false_positives": int(summary_counter["false_positive"]),
        "false_negatives": int(summary_counter["false_negative"]),
        "failure_rate": round_float(safe_div(failed, summary_counter["prediction_rows"])),
    }
    categories = [
        {"category": category, "count": count, "share_of_failures": round_float(safe_div(count, failed))}
        for category, count in sorted(category_counter.items())
    ]
    return {"summary": summary, "categories": categories, "examples": examples}


def classify_failure(error_type: str, profile: dict[str, Any], row: dict[str, Any]) -> list[str]:
    categories: list[str] = []
    score = optional_float(row.get("score"))
    if score is not None and 0.4 <= score <= 0.6:
        categories.append("threshold_margin")
    if profile["behavior_count"] == 0:
        categories.append("no_behavior_record")
    if profile["behavior_count"] <= 2 or profile["mean_consistency"] < 0.4:
        categories.append("weak_behavior_evidence")
    if profile["multi_view_ratio"] < 0.25:
        categories.append("mostly_single_view_support")
    if profile["string_only_ratio"] >= 0.5:
        categories.append("string_dominated_support")
    risky_overlap = len(set(profile["labels"]) & RISK_LABELS)
    if error_type == "false_positive" and risky_overlap >= 3:
        categories.append("benign_utility_overlap")
    if error_type == "false_negative" and risky_overlap <= 1:
        categories.append("low_static_risk_surface")
    if not categories:
        categories.append("mixed_evidence_pattern")
    return categories


def behavior_profile(behavior_doc: dict[str, Any]) -> dict[str, Any]:
    behaviors = list(behavior_doc.get("behaviors", []))
    labels: list[str] = []
    scores: list[float] = []
    multi_view_count = 0
    string_only_count = 0
    top_behaviors: list[dict[str, Any]] = []
    for behavior in behaviors:
        label = str(behavior.get("label", "unknown"))
        labels.append(label)
        score = float(behavior.get("consistency_score", 0.0) or 0.0)
        scores.append(score)
        support = normalize_support(behavior.get("support_by_view", {}))
        nonzero_views = [view for view, count in support.items() if count > 0]
        multi_view_count += int(len(nonzero_views) >= 2)
        string_only_count += int(nonzero_views == ["string"])
        top_behaviors.append(
            {
                "label": label,
                "score": round_float(score),
                "views": ",".join(nonzero_views),
                "evidence_refs": len(behavior.get("evidence_ids", [])),
            }
        )
    top_behaviors.sort(key=lambda item: (item["score"], item["evidence_refs"]), reverse=True)
    return {
        "behavior_count": len(behaviors),
        "labels": labels,
        "mean_consistency": round_float(mean(scores)),
        "multi_view_ratio": round_float(safe_div(multi_view_count, len(behaviors))),
        "string_only_ratio": round_float(safe_div(string_only_count, len(behaviors))),
        "top_behaviors": top_behaviors[:5],
    }


def load_jsonl_by_sample_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        sample_id = row.get("sample_id")
        if sample_id:
            rows[str(sample_id)] = row
    return rows


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return list(iter_jsonl(path))
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = collect_prediction_rows(payload)
        if rows:
            return rows
    raise ValueError(f"No prediction rows found in {path}. Re-run experiments with --save-predictions.")


def collect_prediction_rows(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        prediction_rows = payload.get("prediction_rows")
        if isinstance(prediction_rows, list):
            rows.extend(row for row in prediction_rows if isinstance(row, dict))
        for value in payload.values():
            if isinstance(value, (dict, list)):
                rows.extend(collect_prediction_rows(value))
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, (dict, list)):
                rows.extend(collect_prediction_rows(item))
    return rows


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_support(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(count) for key, count in value.items() if int(count) > 0}


def label_to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "malware", "malicious"}:
        return 1
    if text in {"0", "benign", "goodware"}:
        return 0
    return None


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def new_behavior_stat() -> dict[str, Any]:
    return {
        "behavior_label": "",
        "records": 0,
        "samples": set(),
        "evidence_refs": 0,
        "valid_records": 0,
        "support_match_records": 0,
        "multi_view_records": 0,
        "consistency_scores": [],
        "permission_support": 0,
        "api_support": 0,
        "component_support": 0,
        "string_support": 0,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    quality: dict[str, Any],
    failure_result: dict[str, Any] | None,
    had_predictions: bool,
) -> None:
    summary = quality["summary"]
    lines = [
        "# Evidence Quality and Failure Analysis",
        "",
        "## Evidence Constraint Quality",
        "",
        f"- Behavior records: {summary['behavior_records']}",
        f"- Valid evidence reference rate: {summary['valid_evidence_reference_rate']:.4f}",
        f"- Records with all references valid: {summary['all_refs_valid_record_rate']:.4f}",
        f"- Support-by-view match rate: {summary['support_match_record_rate']:.4f}",
        f"- Multi-view behavior rate: {summary['multi_view_behavior_rate']:.4f}",
        f"- Mean evidence references per behavior: {summary['mean_evidence_refs_per_behavior']:.2f}",
        "",
    ]
    if failure_result is None:
        lines.extend(
            [
                "## Failure Cases",
                "",
                "Per-sample prediction rows were not provided, so failure-case analysis was not run.",
                "Re-run the target experiment with `--save-predictions`, then pass the resulting metric JSON to `--predictions`.",
                "",
            ]
        )
    else:
        failure = failure_result["summary"]
        lines.extend(
            [
                "## Failure Cases",
                "",
                f"- Prediction rows: {failure['prediction_rows']}",
                f"- Failed rows: {failure['failed_rows']}",
                f"- False positives: {failure['false_positives']}",
                f"- False negatives: {failure['false_negatives']}",
                f"- Failure rate: {failure['failure_rate']:.4f}",
                "",
            ]
        )
    if not had_predictions:
        lines.append("Note: failure tables in the paper should remain empty until prediction rows are generated.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean(values: list[float] | list[int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def median(values: list[float] | list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


if __name__ == "__main__":
    main()
