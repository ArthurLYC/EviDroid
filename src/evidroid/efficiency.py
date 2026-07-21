from __future__ import annotations

import math
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from evidroid.analyzers.llm_analyzer import OpenAIBehaviorAnalyzer
from evidroid.classifier_selection import make_classifier_pipeline
from evidroid.extractors.androguard_extractor import AndroguardEvidenceExtractor
from evidroid.features import build_ablation_feature_dict
from evidroid.io_utils import read_jsonl, write_json, write_jsonl
from evidroid.modeling import _adjust_test_size
from evidroid.settings import load_llm_config


def run_efficiency_analysis(
    raw_dir: str | Path = "data/raw",
    evidence_path: str | Path = "data/processed/evidence.jsonl",
    behavior_path: str | Path = "data/processed/behaviors_deepseek.jsonl",
    out_dir: str | Path = "artifacts/efficiency",
    limit_per_class: int | None = None,
    limit: int | None = None,
    measure_extract: bool = False,
    behavior_analyzer: str = "existing",
    llm_config_path: str | Path | None = None,
    model: str | None = None,
    classifier: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    min_consistency: float = 0.0,
    min_support_views: int = 1,
    top_k_behaviors: int | None = None,
    select_k_best: int = 0,
    static_profile: str = "basic",
) -> dict[str, Any]:
    if behavior_analyzer not in {"existing", "llm"}:
        raise ValueError("behavior_analyzer must be one of: existing, llm")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extraction_result = _load_or_measure_extraction(
        raw_dir=raw_dir,
        evidence_path=evidence_path,
        out_dir=out_dir,
        limit_per_class=limit_per_class,
        limit=limit,
        measure_extract=measure_extract,
    )
    evidence_rows = extraction_result["evidence_rows"]

    behavior_result = _load_or_measure_behavior(
        evidence_rows=evidence_rows,
        behavior_path=behavior_path,
        out_dir=out_dir,
        behavior_analyzer=behavior_analyzer,
        llm_config_path=llm_config_path,
        model=model,
    )
    behavior_rows = behavior_result["behavior_rows"]

    feature_result = _measure_feature_construction(
        evidence_rows,
        behavior_rows,
        min_consistency=min_consistency,
        min_support_views=min_support_views,
        top_k_behaviors=top_k_behaviors,
        static_profile=static_profile,
    )
    classification_result = _measure_classifier(
        feature_rows=feature_result["feature_rows"],
        labels=feature_result["labels"],
        sample_ids=feature_result["sample_ids"],
        out_dir=out_dir,
        classifier=classifier,
        test_size=test_size,
        random_state=random_state,
        select_k_best=select_k_best,
    )

    summary = {
        "sample_count": len(evidence_rows),
        "labeled_sample_count": len(feature_result["labels"]),
        "behavior_path": str(behavior_path),
        "evidence_path": str(evidence_path),
        "extraction": extraction_result["summary"],
        "evidence_scale": summarize_evidence_scale(evidence_rows),
        "behavior": behavior_result["summary"],
        "feature_filters": {
            "min_consistency": min_consistency,
            "min_support_views": min_support_views,
            "top_k_behaviors": top_k_behaviors,
            "select_k_best": select_k_best,
            "static_profile": static_profile,
        },
        "feature_construction": feature_result["summary"],
        "classification": classification_result,
    }
    write_json(out_dir / "efficiency_metrics.json", summary)
    _write_markdown_report(out_dir / "efficiency_report.md", summary)
    return summary


def summarize_numbers(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
            "sum": 0.0,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": float(mean(ordered)),
        "median": float(median(ordered)),
        "p95": float(_percentile(ordered, 95)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "sum": float(sum(ordered)),
    }


def summarize_evidence_scale(evidence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    views = ["permission", "api", "component", "string"]
    view_counts: dict[str, list[float]] = {view: [] for view in views}
    total_counts: list[float] = []
    error_count = 0
    for row in evidence_rows:
        counts = row.get("view_counts", {})
        total = 0
        for view in views:
            value = int(counts.get(view, 0))
            view_counts[view].append(float(value))
            total += value
        total_counts.append(float(total))
        if row.get("errors"):
            error_count += 1
    return {
        "total_evidence": summarize_numbers(total_counts),
        "by_view": {view: summarize_numbers(values) for view, values in view_counts.items()},
        "rows_with_errors": error_count,
    }


def _load_or_measure_extraction(
    raw_dir: str | Path,
    evidence_path: str | Path,
    out_dir: Path,
    limit_per_class: int | None,
    limit: int | None,
    measure_extract: bool,
) -> dict[str, Any]:
    if not measure_extract:
        rows = read_jsonl(evidence_path)
        timings = _timings_from_rows(rows, "extract_seconds")
        return {
            "evidence_rows": rows,
            "summary": {
                "mode": "existing_with_timing" if timings else "existing",
                "enabled": bool(timings),
                "seconds": summarize_numbers(timings) if timings else None,
                "timed_sample_count": len(timings),
                "errors": sum(1 for row in rows if row.get("errors")),
            },
        }

    apk_rows = _collect_apks(Path(raw_dir), limit_per_class=limit_per_class, limit=limit)
    extractor = AndroguardEvidenceExtractor()
    evidence_rows: list[dict[str, Any]] = []
    timings: list[float] = []
    for idx, (apk_path, label) in enumerate(apk_rows, start=1):
        print(f"[efficiency:extract] {idx}/{len(apk_rows)} {label} {apk_path.name}")
        start = time.perf_counter()
        doc = extractor.extract(apk_path, label=label)
        elapsed = time.perf_counter() - start
        doc["timing"] = {"extract_seconds": elapsed}
        evidence_rows.append(doc)
        timings.append(elapsed)

    write_jsonl(out_dir / "timed_evidence.jsonl", evidence_rows)
    return {
        "evidence_rows": evidence_rows,
        "summary": {
            "mode": "measured",
            "enabled": True,
            "sample_count": len(evidence_rows),
            "seconds": summarize_numbers(timings),
            "errors": sum(1 for row in evidence_rows if row.get("errors")),
            "output": str(out_dir / "timed_evidence.jsonl"),
        },
    }


def _load_or_measure_behavior(
    evidence_rows: list[dict[str, Any]],
    behavior_path: str | Path,
    out_dir: Path,
    behavior_analyzer: str,
    llm_config_path: str | Path | None,
    model: str | None,
) -> dict[str, Any]:
    if behavior_analyzer == "existing":
        rows = read_jsonl(behavior_path)
        return {
            "behavior_rows": rows,
            "summary": _behavior_summary(rows, mode="existing", timings=None),
        }

    config = load_llm_config(llm_config_path or "configs/deepseek.json")
    if model:
        config["model"] = model
    analyzer = OpenAIBehaviorAnalyzer.from_config(config)

    rows: list[dict[str, Any]] = []
    timings: list[float] = []
    for idx, evidence_doc in enumerate(evidence_rows, start=1):
        print(f"[efficiency:behavior] {idx}/{len(evidence_rows)} {evidence_doc['sample_id']}")
        start = time.perf_counter()
        behavior_doc = analyzer.analyze(evidence_doc)
        elapsed = time.perf_counter() - start
        behavior_doc["timing"] = {"behavior_seconds": elapsed}
        rows.append(behavior_doc)
        timings.append(elapsed)

    output = out_dir / f"timed_behaviors_{behavior_analyzer}.jsonl"
    write_jsonl(output, rows)
    summary = _behavior_summary(rows, mode=behavior_analyzer, timings=timings)
    summary["output"] = str(output)
    return {"behavior_rows": rows, "summary": summary}


def _behavior_summary(
    behavior_rows: list[dict[str, Any]],
    mode: str,
    timings: list[float] | None,
) -> dict[str, Any]:
    behavior_counts = [float(len(row.get("behaviors", []))) for row in behavior_rows]
    effective_timings = timings if timings is not None else _timings_from_rows(behavior_rows, "behavior_seconds")
    usage_rows = [row.get("usage", {}) for row in behavior_rows if row.get("usage")]
    usage_summary = None
    if usage_rows:
        usage_summary = {
            "prompt_tokens": summarize_numbers([float(row.get("prompt_tokens") or 0) for row in usage_rows]),
            "completion_tokens": summarize_numbers([float(row.get("completion_tokens") or 0) for row in usage_rows]),
            "total_tokens": summarize_numbers([float(row.get("total_tokens") or 0) for row in usage_rows]),
        }
    return {
        "mode": mode,
        "enabled": bool(effective_timings),
        "sample_count": len(behavior_rows),
        "seconds": summarize_numbers(effective_timings) if effective_timings else None,
        "timed_sample_count": len(effective_timings),
        "behavior_count": summarize_numbers(behavior_counts),
        "usage": usage_summary,
    }


def _timings_from_rows(rows: list[dict[str, Any]], key: str) -> list[float]:
    timings: list[float] = []
    for row in rows:
        timing = row.get("timing")
        if not isinstance(timing, dict):
            continue
        value = timing.get(key)
        if isinstance(value, (int, float)):
            timings.append(float(value))
    return timings


def _measure_feature_construction(
    evidence_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    min_consistency: float = 0.0,
    min_support_views: int = 1,
    top_k_behaviors: int | None = None,
    static_profile: str = "basic",
) -> dict[str, Any]:
    behavior_by_id = {row["sample_id"]: row for row in behavior_rows}
    feature_rows: list[dict[str, float]] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    timings: list[float] = []
    feature_counts: list[float] = []

    for evidence_doc in evidence_rows:
        label = evidence_doc.get("label")
        if label not in {"benign", "malware"}:
            continue
        sample_id = evidence_doc["sample_id"]
        start = time.perf_counter()
        features = build_ablation_feature_dict(
            evidence_doc,
            behavior_by_id.get(sample_id, {"sample_id": sample_id, "behaviors": []}),
            use_behavior_semantics=True,
            use_consistency=True,
            min_consistency=min_consistency,
            min_support_views=min_support_views,
            top_k_behaviors=top_k_behaviors,
            static_profile=static_profile,
        )
        elapsed = time.perf_counter() - start
        feature_rows.append(features)
        labels.append(1 if label == "malware" else 0)
        sample_ids.append(sample_id)
        timings.append(elapsed)
        feature_counts.append(float(len(features)))

    return {
        "feature_rows": feature_rows,
        "labels": labels,
        "sample_ids": sample_ids,
        "summary": {
            "feature_set": "A3_full_evidroid",
            "seconds": summarize_numbers(timings),
            "feature_count_per_sample": summarize_numbers(feature_counts),
        },
    }


def _measure_classifier(
    feature_rows: list[dict[str, float]],
    labels: list[int],
    sample_ids: list[str],
    out_dir: Path,
    classifier: str,
    test_size: float,
    random_state: int,
    select_k_best: int = 0,
) -> dict[str, Any]:
    if len(set(labels)) < 2:
        raise ValueError("Need at least two classes for classifier timing.")
    if len(labels) < 4:
        return {
            "classifier": classifier,
            "status": "skipped",
            "reason": "Need at least four labeled samples for a train/test split.",
            "sample_count": len(labels),
        }
    test_size = _adjust_test_size(test_size, len(labels), len(set(labels)))
    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        feature_rows,
        labels,
        sample_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    model = make_classifier_pipeline(
        classifier,
        random_state=random_state,
        select_k_best=select_k_best,
    )

    fit_start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    predictions = model.predict(x_test)
    scores = _decision_scores(model, x_test)
    predict_seconds = time.perf_counter() - predict_start

    model_path = out_dir / f"{classifier}_efficiency_model.joblib"
    joblib.dump(model, model_path)

    result: dict[str, Any] = {
        "classifier": classifier,
        "train_sample_count": len(x_train),
        "test_sample_count": len(x_test),
        "train_sample_ids": id_train,
        "test_sample_ids": id_test,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "avg_predict_seconds_per_sample": float(predict_seconds / max(1, len(x_test))),
        "model_path": str(model_path),
        "model_size_bytes": int(model_path.stat().st_size),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(y_test, scores))
    except ValueError:
        result["roc_auc"] = None
    return result


def _decision_scores(model: Any, x_rows: list[dict[str, float]]) -> list[float]:
    classifier = model.named_steps["classifier"]
    transformed = model[:-1].transform(x_rows)
    if hasattr(classifier, "predict_proba"):
        return [float(row[1]) for row in classifier.predict_proba(transformed)]
    if hasattr(classifier, "decision_function"):
        raw_scores = classifier.decision_function(transformed)
        return [float(value) for value in raw_scores]
    return [float(value) for value in classifier.predict(transformed)]


def _collect_apks(
    raw_dir: Path,
    limit_per_class: int | None = None,
    limit: int | None = None,
) -> list[tuple[Path, str | None]]:
    rows: list[tuple[Path, str | None]] = []
    class_dirs = [(raw_dir / "benign", "benign"), (raw_dir / "malware", "malware")]
    if all(path.exists() for path, _label in class_dirs):
        for path, label in class_dirs:
            files = sorted(path.rglob("*.apk"))
            if limit_per_class is not None:
                files = files[:limit_per_class]
            rows.extend((item, label) for item in files)
    else:
        files = sorted(raw_dir.rglob("*.apk"))
        if limit is not None:
            files = files[:limit]
        rows.extend((item, None) for item in files)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    extraction = summary["extraction"]
    behavior = summary["behavior"]
    feature = summary["feature_construction"]
    classification = summary["classification"]
    evidence_scale = summary["evidence_scale"]

    classification_ok = classification.get("status", "ok") != "skipped"
    lines = [
        "# Efficiency Analysis",
        "",
        "## Overview",
        "",
        f"- Sample count: `{summary['sample_count']}`",
        f"- Labeled sample count: `{summary['labeled_sample_count']}`",
        f"- Behavior mode: `{behavior['mode']}`",
        f"- Classifier: `{classification['classifier']}`",
        "",
        "## Runtime",
        "",
        "| Stage | Enabled | Mean seconds | Median seconds | P95 seconds | Total seconds |",
        "|---|---:|---:|---:|---:|---:|",
        _runtime_row("APK evidence extraction", extraction.get("enabled"), extraction.get("seconds")),
        _runtime_row("Behavior inference", behavior.get("enabled"), behavior.get("seconds")),
        _runtime_row("Feature construction", True, feature["seconds"]),
        _classification_runtime_row("Classifier training", classification_ok, classification.get("fit_seconds")),
        _classification_runtime_row("Classifier prediction", classification_ok, classification.get("predict_seconds")),
        "",
        "## Evidence Scale",
        "",
        "| View | Mean count | Median count | P95 count | Max count |",
        "|---|---:|---:|---:|---:|",
    ]
    for view, stats in evidence_scale["by_view"].items():
        lines.append(
            f"| {view} | {_fmt(stats['mean'], 2)} | {_fmt(stats['median'], 2)} | "
            f"{_fmt(stats['p95'], 2)} | {_fmt(stats['max'], 0)} |"
        )
    lines.extend(
        [
            "",
            "## Classification",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Status | {classification.get('status', 'ok')} |",
            f"| Accuracy | {_fmt(classification.get('accuracy'))} |",
            f"| Precision | {_fmt(classification.get('precision'))} |",
            f"| Recall | {_fmt(classification.get('recall'))} |",
            f"| F1 | {_fmt(classification.get('f1'))} |",
            f"| ROC-AUC | {_fmt(classification.get('roc_auc'))} |",
            f"| Model size bytes | {classification.get('model_size_bytes', '-')} |",
        ]
    )
    usage = behavior.get("usage")
    if usage:
        lines.extend(
            [
                "",
                "## Token Usage",
                "",
                "| Token type | Mean | Total |",
                "|---|---:|---:|",
                f"| Prompt | {_fmt(usage['prompt_tokens']['mean'], 2)} | {_fmt(usage['prompt_tokens']['sum'], 0)} |",
                f"| Completion | {_fmt(usage['completion_tokens']['mean'], 2)} | {_fmt(usage['completion_tokens']['sum'], 0)} |",
                f"| Total | {_fmt(usage['total_tokens']['mean'], 2)} | {_fmt(usage['total_tokens']['sum'], 0)} |",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _runtime_row(stage: str, enabled: bool | None, stats: dict[str, Any] | None) -> str:
    mark = "yes" if enabled else "no"
    if not stats:
        return f"| {stage} | {mark} | - | - | - | - |"
    return (
        f"| {stage} | {mark} | {_fmt(stats['mean'])} | {_fmt(stats['median'])} | "
        f"{_fmt(stats['p95'])} | {_fmt(stats['sum'])} |"
    )


def _classification_runtime_row(stage: str, enabled: bool, seconds: float | None) -> str:
    mark = "yes" if enabled else "no"
    if seconds is None:
        return f"| {stage} | {mark} | - | - | - | - |"
    return f"| {stage} | {mark} | {_fmt(seconds)} | {_fmt(seconds)} | {_fmt(seconds)} | {_fmt(seconds)} |"
