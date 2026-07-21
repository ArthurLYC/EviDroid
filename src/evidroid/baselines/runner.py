from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

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

from evidroid.baselines.static import abstract_api
from evidroid.classifier_selection import feature_selection_metadata, make_classifier_pipeline
from evidroid.io_utils import read_jsonl

SAMPLE_ID_RE = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')
LABEL_RE = re.compile(r'"label"\s*:\s*"(benign|malware)"')


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_sample_index(evidence_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with evidence_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sample_match = SAMPLE_ID_RE.search(line)
            label_match = LABEL_RE.search(line)
            if not sample_match or not label_match:
                continue
            label = label_match.group(1)
            rows.append({"sample_id": sample_match.group(1), "label": label, "label_int": 1 if label == "malware" else 0})
    if not rows:
        for row in iter_jsonl(evidence_path):
            label = row.get("label")
            if label not in {"benign", "malware"}:
                continue
            rows.append({"sample_id": row["sample_id"], "label": label, "label_int": 1 if label == "malware" else 0})
    return rows


def run_streamed_sklearn_method(
    evidence_path: Path,
    train_ids: list[str],
    test_ids: list[str],
    y_train: list[int],
    y_test: list[int],
    feature_builder: Any,
    classifier: str,
    select_k_best: int,
    random_state: int,
    out_dir: Path,
    model_name: str,
    display_name: str,
    feature_type: str,
) -> dict[str, Any]:
    wanted = set(train_ids) | set(test_ids)
    features_by_id: dict[str, dict[str, float]] = {}
    start = time.perf_counter()
    for _idx, evidence_doc in enumerate(iter_jsonl(evidence_path), start=1):
        sample_id = evidence_doc.get("sample_id")
        if sample_id not in wanted:
            continue
        features_by_id[sample_id] = feature_builder(evidence_doc)
        if len(features_by_id) % 5000 == 0:
            print(f"[{model_name}] built {len(features_by_id)} feature rows", flush=True)
    build_seconds = time.perf_counter() - start
    x_train = [features_by_id[sample_id] for sample_id in train_ids]
    x_test = [features_by_id[sample_id] for sample_id in test_ids]
    metrics = train_sklearn_features(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        test_ids=test_ids,
        classifier=classifier,
        select_k_best=select_k_best,
        random_state=random_state,
        out_dir=out_dir,
        model_name=model_name,
        display_name=display_name,
        feature_type=feature_type,
    )
    metrics["feature_build_seconds"] = float(build_seconds)
    return metrics


def run_prebuilt_sklearn_method(
    feature_by_id: dict[str, dict[str, float]],
    train_ids: list[str],
    test_ids: list[str],
    y_train: list[int],
    y_test: list[int],
    classifier: str,
    select_k_best: int,
    random_state: int,
    out_dir: Path,
    model_name: str,
    display_name: str,
    feature_type: str,
) -> dict[str, Any]:
    x_train = [feature_by_id.get(sample_id, {}) for sample_id in train_ids]
    x_test = [feature_by_id.get(sample_id, {}) for sample_id in test_ids]
    return train_sklearn_features(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        test_ids=test_ids,
        classifier=classifier,
        select_k_best=select_k_best,
        random_state=random_state,
        out_dir=out_dir,
        model_name=model_name,
        display_name=display_name,
        feature_type=feature_type,
    )


def train_sklearn_features(
    x_train: list[dict[str, float]],
    x_test: list[dict[str, float]],
    y_train: list[int],
    y_test: list[int],
    test_ids: list[str],
    classifier: str,
    select_k_best: int,
    random_state: int,
    out_dir: Path,
    model_name: str,
    display_name: str,
    feature_type: str,
) -> dict[str, Any]:
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
    scores = decision_scores(model, x_test)
    predict_seconds = time.perf_counter() - predict_start
    model_path = out_dir / f"{model_name}_model.joblib"
    joblib.dump(model, model_path)
    return evaluate_predictions(
        name=model_name,
        display_name=display_name,
        feature_type=feature_type,
        classifier=classifier,
        y_test=y_test,
        predictions=predictions,
        scores=scores,
        test_ids=test_ids,
        extra={
            "status": "ok",
            **feature_selection_metadata(model),
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "model_path": str(model_path),
            "model_size_bytes": int(model_path.stat().st_size),
        },
    )


def decision_scores(model: Any, x_rows: list[dict[str, float]]) -> list[float]:
    classifier = model.named_steps["classifier"]
    transformed = model[:-1].transform(x_rows)
    if hasattr(classifier, "predict_proba"):
        return [float(item[1]) for item in classifier.predict_proba(transformed)]
    if hasattr(classifier, "decision_function"):
        return [float(item) for item in classifier.decision_function(transformed)]
    return [float(item) for item in classifier.predict(transformed)]


def load_mamadroid_cache(cache_path: Path) -> dict[str, dict[str, float]]:
    rows = read_jsonl(cache_path)
    return {row["sample_id"]: dict(row.get("features", {})) for row in rows if "features" in row}


def convert_mamadroid_package_cache_to_family(
    feature_by_id: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        sample_id: convert_mamadroid_package_features_to_family(features)
        for sample_id, features in feature_by_id.items()
    }


def convert_mamadroid_package_features_to_family(features: dict[str, float]) -> dict[str, float]:
    prefix = "mamadroid::"
    outgoing: dict[str, float] = {}
    transitions: dict[tuple[str, str], float] = {}
    for key, value in features.items():
        if not key.startswith(prefix) or "->" not in key:
            continue
        source, target = key[len(prefix) :].split("->", 1)
        family_source = mamadroid_node_to_family(source)
        family_target = mamadroid_node_to_family(target)
        transition_key = (family_source, family_target)
        transitions[transition_key] = transitions.get(transition_key, 0.0) + float(value)
        outgoing[family_source] = outgoing.get(family_source, 0.0) + float(value)

    family_features: dict[str, float] = {}
    for (source, target), value in transitions.items():
        denominator = outgoing.get(source, 0.0)
        family_features[f"{prefix}{source}->{target}"] = value / denominator if denominator else value
    return family_features


def mamadroid_node_to_family(node: str) -> str:
    if node in {"unknown", "self_defined"}:
        return node
    return abstract_api(node, abstraction="family")


def evaluate_predictions(
    name: str,
    display_name: str,
    feature_type: str,
    classifier: str,
    y_test: list[int],
    predictions: Any,
    scores: list[float],
    test_ids: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "name": name,
        "display_name": display_name,
        "feature_type": feature_type,
        "classifier": classifier,
        "test_sample_ids": test_ids,
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
    if extra:
        metrics.update(extra)
    return metrics
