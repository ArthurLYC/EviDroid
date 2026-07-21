from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from evidroid.classifier_selection import feature_selection_metadata, make_classifier_pipeline
from evidroid.features import build_ablation_feature_dict
from evidroid.io_utils import read_jsonl, write_json
from evidroid.modeling import _adjust_test_size


ABLATION_VARIANTS = [
    {
        "id": "A0",
        "name": "Static",
        "use_behavior_semantics": False,
        "use_consistency": False,
        "description": "Static evidence features only.",
    },
    {
        "id": "A1",
        "name": "Static+Behavior",
        "use_behavior_semantics": True,
        "use_consistency": False,
        "description": "Static evidence plus evidence-constrained behavior semantic labels.",
    },
    {
        "id": "A2",
        "name": "Static+Consistency",
        "use_behavior_semantics": False,
        "use_consistency": True,
        "description": "Static evidence plus dynamic cross-view consistency statistics, without independent behavior semantic features.",
    },
    {
        "id": "A3",
        "name": "Full EviDroid",
        "use_behavior_semantics": True,
        "use_consistency": True,
        "description": "Static evidence plus behavior semantic labels and dynamic cross-view consistency features.",
    },
]


def run_ablation(
    evidence_path: str | Path,
    behavior_path: str | Path,
    out_dir: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
    classifier: str = "random_forest",
    min_consistency: float = 0.0,
    min_support_views: int = 1,
    top_k_behaviors: int | None = None,
    select_k_best: int = 0,
    static_profile: str = "basic",
    feature_version: str = "v1",
    grouped_select_k_best: dict[str, int] | None = None,
) -> dict[str, Any]:
    evidence_rows = [
        row
        for row in read_jsonl(evidence_path)
        if row.get("label") in {"benign", "malware"}
    ]
    if not evidence_rows:
        raise ValueError(f"No labeled evidence rows found in {evidence_path}.")

    behavior_rows = {row["sample_id"]: row for row in read_jsonl(behavior_path)}
    sample_ids = [row["sample_id"] for row in evidence_rows]
    y = [1 if row["label"] == "malware" else 0 for row in evidence_rows]
    if len(set(y)) < 2:
        raise ValueError("Need at least two classes for ablation.")
    test_size = _adjust_test_size(test_size, len(y), len(set(y)))

    train_ids, test_ids, y_train, y_test = train_test_split(
        sample_ids,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    evidence_by_id = {row["sample_id"]: row for row in evidence_rows}
    split = {
        "train_sample_ids": train_ids,
        "test_sample_ids": test_ids,
        "random_state": random_state,
        "test_size": test_size,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "split.json", split)

    metrics_rows: list[dict[str, Any]] = []
    for variant in ABLATION_VARIANTS:
        x_train = [
            _build_variant_features(
                evidence_by_id[sample_id],
                behavior_rows,
                variant,
                min_consistency=min_consistency,
                min_support_views=min_support_views,
                top_k_behaviors=top_k_behaviors,
                static_profile=static_profile,
                feature_version=feature_version,
            )
            for sample_id in train_ids
        ]
        x_test = [
            _build_variant_features(
                evidence_by_id[sample_id],
                behavior_rows,
                variant,
                min_consistency=min_consistency,
                min_support_views=min_support_views,
                top_k_behaviors=top_k_behaviors,
                static_profile=static_profile,
                feature_version=feature_version,
            )
            for sample_id in test_ids
        ]
        model = make_classifier_pipeline(
            classifier,
            random_state=random_state,
            select_k_best=0 if grouped_select_k_best else select_k_best,
            grouped_select_k_best=grouped_select_k_best,
        )
        model.fit(x_train, y_train)
        metrics = _evaluate_variant(variant, model, x_test, y_test, test_ids)
        metrics_rows.append(metrics)
        joblib.dump(model, out_dir / f"{variant['id']}_{variant['name'].lower().replace('+', '_').replace(' ', '_')}.joblib")
        write_json(out_dir / f"{variant['id']}_metrics.json", metrics)

    summary = {
        "sample_count": len(sample_ids),
        "behavior_path": str(behavior_path),
        "classifier": classifier,
        "feature_filters": {
            "min_consistency": min_consistency,
            "min_support_views": min_support_views,
            "top_k_behaviors": top_k_behaviors,
            "select_k_best": select_k_best,
            "static_profile": static_profile,
            "feature_version": feature_version,
            "grouped_select_k_best": grouped_select_k_best,
        },
        "split": split,
        "variants": ABLATION_VARIANTS,
        "metrics": metrics_rows,
    }
    write_json(out_dir / "ablation_metrics.json", summary)
    return summary


def _build_variant_features(
    evidence_doc: dict[str, Any],
    behavior_rows: dict[str, dict[str, Any]],
    variant: dict[str, Any],
    min_consistency: float = 0.0,
    min_support_views: int = 1,
    top_k_behaviors: int | None = None,
    static_profile: str = "basic",
    feature_version: str = "v1",
) -> dict[str, float]:
    sample_id = evidence_doc["sample_id"]
    behavior_doc = behavior_rows.get(sample_id, {"sample_id": sample_id, "behaviors": []})
    return build_ablation_feature_dict(
        evidence_doc,
        behavior_doc,
        use_behavior_semantics=bool(variant["use_behavior_semantics"]),
        use_consistency=bool(variant["use_consistency"]),
        min_consistency=min_consistency,
        min_support_views=min_support_views,
        top_k_behaviors=top_k_behaviors,
        static_profile=static_profile,
        feature_version=feature_version,
    )


def _evaluate_variant(
    variant: dict[str, Any],
    model: Pipeline,
    x_test: list[dict[str, float]],
    y_test: list[int],
    test_sample_ids: list[str],
) -> dict[str, Any]:
    predictions = model.predict(x_test)
    scores = _decision_scores(model, x_test)
    metrics: dict[str, Any] = {
        "variant_id": variant["id"],
        "variant_name": variant["name"],
        "use_behavior_semantics": bool(variant["use_behavior_semantics"]),
        "use_consistency": bool(variant["use_consistency"]),
        "description": variant["description"],
        **feature_selection_metadata(model),
        "test_sample_ids": test_sample_ids,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["benign", "malware"],
            zero_division=0,
            output_dict=True,
        ),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, scores))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics


def _decision_scores(model: Pipeline, x_rows: list[dict[str, float]]) -> list[float]:
    classifier = model.named_steps["classifier"]
    transformed = model[:-1].transform(x_rows)
    if hasattr(classifier, "predict_proba"):
        return [float(item[1]) for item in classifier.predict_proba(transformed)]
    if hasattr(classifier, "decision_function"):
        raw_scores = classifier.decision_function(transformed)
        return [float(item) for item in raw_scores]
    return [float(item) for item in classifier.predict(transformed)]
