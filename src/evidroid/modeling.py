from __future__ import annotations

from pathlib import Path
from math import ceil
from typing import Any

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
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

from evidroid.features import build_feature_dict
from evidroid.io_utils import read_jsonl, write_json


def train_and_evaluate(
    evidence_path: str | Path,
    behavior_path: str | Path,
    out_dir: str | Path,
    mode: str = "fusion",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    x_rows, y, sample_ids = load_dataset(evidence_path, behavior_path, mode=mode)
    if len(set(y)) < 2:
        raise ValueError("Need at least two classes to train a malware classifier.")
    if len(y) < 4:
        raise ValueError("Need at least four samples for a train/test split.")
    test_size = _adjust_test_size(test_size, len(y), len(set(y)))

    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        x_rows,
        y,
        sample_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics: dict[str, Any] = {
        "mode": mode,
        "sample_count": int(len(y)),
        "feature_count": int(len(model.named_steps["vectorizer"].vocabulary_)),
        "test_sample_ids": id_test,
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
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    except ValueError:
        metrics["roc_auc"] = None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / f"{mode}_model.joblib")
    write_json(out_dir / f"{mode}_metrics.json", metrics)
    return metrics


def load_dataset(
    evidence_path: str | Path,
    behavior_path: str | Path,
    mode: str = "fusion",
) -> tuple[list[dict[str, float]], list[int], list[str]]:
    evidence_rows = read_jsonl(evidence_path)
    behavior_rows = {row["sample_id"]: row for row in read_jsonl(behavior_path)}
    x_rows: list[dict[str, float]] = []
    y: list[int] = []
    sample_ids: list[str] = []

    for evidence_doc in evidence_rows:
        label = evidence_doc.get("label")
        if label not in {"benign", "malware"}:
            continue
        sample_id = evidence_doc["sample_id"]
        behavior_doc = behavior_rows.get(sample_id, {"sample_id": sample_id, "behaviors": []})
        x_rows.append(build_feature_dict(evidence_doc, behavior_doc, mode=mode))
        y.append(1 if label == "malware" else 0)
        sample_ids.append(sample_id)

    return x_rows, y, sample_ids


def _adjust_test_size(test_size: float, sample_count: int, class_count: int) -> float:
    if not 0 < test_size < 1:
        return test_size
    requested = ceil(sample_count * test_size)
    if requested >= class_count:
        return test_size
    return class_count / sample_count
