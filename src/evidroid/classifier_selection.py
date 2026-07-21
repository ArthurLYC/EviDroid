from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from scipy import sparse

from evidroid.features import build_evidroid_feature_dict
from evidroid.io_utils import read_jsonl, write_json
from evidroid.modeling import _adjust_test_size


DEFAULT_CLASSIFIERS = [
    "logistic_regression",
    "logistic_regression_sgd",
    "linear_svm",
    "linear_svm_sgd",
    "knn",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "extra_trees_regularized",
    "extra_trees_shallow",
    "extra_trees_deep",
    "extra_trees_calibrated",
    "gradient_boosting",
    "adaboost",
    "xgboost",
    "xgboost_regularized",
    "xgboost_stump",
    "xgboost_shallow",
    "xgboost_depth2",
    "xgboost_compact",
    "xgboost_calibrated",
    "mlp",
    "gaussian_naive_bayes",
    "naive_bayes",
]

CLASSIFIER_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "logistic_regression_sgd": "Logistic Regression (SGD)",
    "linear_svm": "Linear SVM",
    "linear_svm_sgd": "Linear SVM (SGD)",
    "knn": "k-Nearest Neighbors",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "extra_trees_regularized": "Regularized Extra Trees",
    "extra_trees_shallow": "Shallow Extra Trees",
    "extra_trees_deep": "Deep Extra Trees",
    "extra_trees_calibrated": "Calibrated Extra Trees",
    "gradient_boosting": "Gradient Boosting",
    "adaboost": "AdaBoost",
    "xgboost": "XGBoost",
    "xgboost_regularized": "Regularized XGBoost",
    "xgboost_stump": "XGBoost Stump",
    "xgboost_shallow": "Shallow XGBoost",
    "xgboost_depth2": "Depth-2 XGBoost",
    "xgboost_compact": "Compact XGBoost",
    "xgboost_calibrated": "Calibrated XGBoost",
    "lightgbm": "LightGBM",
    "mlp": "MLP",
    "gaussian_naive_bayes": "Gaussian Naive Bayes",
    "naive_bayes": "Complement Naive Bayes",
}


def run_classifier_selection(
    evidence_path: str | Path,
    behavior_path: str | Path,
    out_dir: str | Path,
    classifiers: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    min_consistency: float = 0.0,
    min_support_views: int = 1,
    top_k_behaviors: int | None = None,
    select_k_best: int = 0,
    static_profile: str = "basic",
) -> dict[str, Any]:
    classifiers = classifiers or DEFAULT_CLASSIFIERS
    unknown = sorted(set(classifiers) - set(CLASSIFIER_DISPLAY_NAMES))
    if unknown:
        raise ValueError(f"Unknown classifiers: {unknown}")

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
        raise ValueError("Need at least two classes for classifier selection.")
    test_size = _adjust_test_size(test_size, len(y), len(set(y)))

    evidence_by_id = {row["sample_id"]: row for row in evidence_rows}
    x_rows_by_id = {
        sample_id: build_evidroid_feature_dict(
            evidence_by_id[sample_id],
            behavior_rows.get(sample_id, {"sample_id": sample_id, "behaviors": []}),
            min_consistency=min_consistency,
            min_support_views=min_support_views,
            top_k_behaviors=top_k_behaviors,
            static_profile=static_profile,
        )
        for sample_id in sample_ids
    }

    train_ids, test_ids, y_train, y_test = train_test_split(
        sample_ids,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    x_train = [x_rows_by_id[sample_id] for sample_id in train_ids]
    x_test = [x_rows_by_id[sample_id] for sample_id in test_ids]

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
    for classifier_name in classifiers:
        print(f"[select-classifier] training {classifier_name}")
        metrics = _run_one_classifier(
            classifier_name,
            x_train,
            x_test,
            y_train,
            y_test,
            test_ids,
            out_dir,
            random_state,
            select_k_best,
        )
        metrics_rows.append(metrics)
        write_json(out_dir / f"{classifier_name}_metrics.json", metrics)

    summary = {
        "sample_count": len(sample_ids),
        "behavior_path": str(behavior_path),
        "feature_set": "A3_full_evidroid",
        "feature_filters": {
            "min_consistency": min_consistency,
            "min_support_views": min_support_views,
            "top_k_behaviors": top_k_behaviors,
            "select_k_best": select_k_best,
            "static_profile": static_profile,
        },
        "classifiers": classifiers,
        "split": split,
        "metrics": metrics_rows,
        "best_by_f1": _best_classifier(metrics_rows, key="f1"),
        "best_by_roc_auc": _best_classifier(metrics_rows, key="roc_auc"),
    }
    write_json(out_dir / "classifier_selection_metrics.json", summary)
    return summary


def available_classifiers() -> dict[str, bool]:
    return {
        "logistic_regression": True,
        "logistic_regression_sgd": True,
        "linear_svm": True,
        "linear_svm_sgd": True,
        "knn": True,
        "decision_tree": True,
        "random_forest": True,
        "extra_trees": True,
        "extra_trees_regularized": True,
        "extra_trees_shallow": True,
        "extra_trees_deep": True,
        "extra_trees_calibrated": True,
        "gradient_boosting": True,
        "adaboost": True,
        "xgboost": importlib.util.find_spec("xgboost") is not None,
        "xgboost_regularized": importlib.util.find_spec("xgboost") is not None,
        "xgboost_stump": importlib.util.find_spec("xgboost") is not None,
        "xgboost_shallow": importlib.util.find_spec("xgboost") is not None,
        "xgboost_depth2": importlib.util.find_spec("xgboost") is not None,
        "xgboost_compact": importlib.util.find_spec("xgboost") is not None,
        "xgboost_calibrated": importlib.util.find_spec("xgboost") is not None,
        "lightgbm": importlib.util.find_spec("lightgbm") is not None,
        "mlp": True,
        "gaussian_naive_bayes": True,
        "naive_bayes": True,
    }


def _run_one_classifier(
    classifier_name: str,
    x_train: list[dict[str, float]],
    x_test: list[dict[str, float]],
    y_train: list[int],
    y_test: list[int],
    test_sample_ids: list[str],
    out_dir: Path,
    random_state: int,
    select_k_best: int = 0,
) -> dict[str, Any]:
    if not available_classifiers().get(classifier_name, False):
        return {
            "classifier": classifier_name,
            "display_name": CLASSIFIER_DISPLAY_NAMES[classifier_name],
            "status": "skipped",
            "reason": f"Optional package for {classifier_name} is not installed.",
        }

    try:
        model = make_classifier_pipeline(
            classifier_name,
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

        model_path = out_dir / f"{classifier_name}_model.joblib"
        joblib.dump(model, model_path)
        return _evaluate_classifier(
            classifier_name=classifier_name,
            model=model,
            model_path=model_path,
            y_test=y_test,
            predictions=predictions,
            scores=scores,
            test_sample_ids=test_sample_ids,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
        )
    except Exception as exc:
        return {
            "classifier": classifier_name,
            "display_name": CLASSIFIER_DISPLAY_NAMES[classifier_name],
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def make_classifier_pipeline(
    classifier_name: str,
    random_state: int = 42,
    select_k_best: int = 0,
    grouped_select_k_best: dict[str, int] | None = None,
) -> Pipeline:
    if grouped_select_k_best:
        shared_steps: list[tuple[str, Any]] = [
            ("vectorizer", GroupedDictVectorizerSelector(group_k=grouped_select_k_best))
        ]
    else:
        shared_steps = [("vectorizer", DictVectorizer(sparse=True))]
    if select_k_best and select_k_best > 0 and not grouped_select_k_best:
        shared_steps.append(("select", CappedSelectKBest(k=select_k_best)))
    if classifier_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "linear_svm":
        classifier = LinearSVC(
            class_weight="balanced",
            random_state=random_state,
            dual=False,
            max_iter=10_000,
            tol=1e-3,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "logistic_regression_sgd":
        classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            random_state=random_state,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "linear_svm_sgd":
        classifier = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-5,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            random_state=random_state,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "knn":
        classifier = KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "decision_tree":
        classifier = DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "extra_trees":
        classifier = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "extra_trees_regularized":
        classifier = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "extra_trees_shallow":
        classifier = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=30,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "extra_trees_deep":
        classifier = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "extra_trees_calibrated":
        base_classifier = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        classifier = CalibratedClassifierCV(
            estimator=base_classifier,
            method="sigmoid",
            cv=3,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.05,
            min_samples_leaf=20,
            random_state=random_state,
        )
        steps = [*shared_steps, ("dense", FunctionTransformer(_to_dense, accept_sparse=True)), ("classifier", classifier)]
    elif classifier_name == "adaboost":
        classifier = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, min_samples_leaf=20, random_state=random_state),
            n_estimators=120,
            learning_rate=0.05,
            random_state=random_state,
        )
        steps = [*shared_steps, ("dense", FunctionTransformer(_to_dense, accept_sparse=True)), ("classifier", classifier)]
    elif classifier_name == "xgboost":
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "xgboost_regularized":
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            n_estimators=600,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=3.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=3.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "xgboost_stump":
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            n_estimators=220,
            max_depth=1,
            learning_rate=0.04,
            min_child_weight=2.0,
            subsample=0.95,
            colsample_bytree=1.0,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "xgboost_shallow":
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            n_estimators=360,
            max_depth=2,
            learning_rate=0.035,
            min_child_weight=3.0,
            subsample=0.9,
            colsample_bytree=1.0,
            reg_lambda=2.5,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "xgboost_depth2":
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            n_estimators=260,
            max_depth=2,
            learning_rate=0.06,
            min_child_weight=1.0,
            subsample=0.95,
            colsample_bytree=0.95,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "xgboost_compact":
        from xgboost import XGBClassifier

        classifier = XGBClassifier(
            n_estimators=120,
            max_depth=2,
            learning_rate=0.08,
            min_child_weight=4.0,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=3.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "xgboost_calibrated":
        from xgboost import XGBClassifier

        base_classifier = XGBClassifier(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.05,
            min_child_weight=2.0,
            subsample=0.95,
            colsample_bytree=1.0,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        classifier = CalibratedClassifierCV(
            estimator=base_classifier,
            method="sigmoid",
            cv=3,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "lightgbm":
        from lightgbm import LGBMClassifier

        classifier = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        steps = [*shared_steps, ("classifier", classifier)]
    elif classifier_name == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            early_stopping=False,
            random_state=random_state,
        )
        steps = [
            *shared_steps,
            ("dense", FunctionTransformer(_to_dense, accept_sparse=True)),
            ("classifier", classifier),
        ]
    elif classifier_name == "gaussian_naive_bayes":
        classifier = GaussianNB()
        steps = [
            *shared_steps,
            ("dense", FunctionTransformer(_to_dense, accept_sparse=True)),
            ("classifier", classifier),
        ]
    elif classifier_name == "naive_bayes":
        classifier = ComplementNB()
        steps = [*shared_steps, ("classifier", classifier)]
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    return Pipeline(steps=steps)


class CappedSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 0) -> None:
        self.k = k

    def fit(self, x_matrix: Any, y: Any = None) -> "CappedSelectKBest":
        n_features = int(getattr(x_matrix, "shape", [0, 0])[1])
        if self.k <= 0 or n_features <= self.k:
            self.selector_ = None
            self.selected_feature_count_ = n_features
            return self
        self.selector_ = SelectKBest(score_func=chi2, k=int(self.k))
        self.selector_.fit(x_matrix, y)
        self.selected_feature_count_ = int(self.k)
        return self

    def transform(self, x_matrix: Any) -> Any:
        if getattr(self, "selector_", None) is None:
            return x_matrix
        return self.selector_.transform(x_matrix)


class GroupedDictVectorizerSelector(BaseEstimator, TransformerMixin):
    """Vectorize feature dicts and apply per-family feature selection."""

    def __init__(self, group_k: dict[str, int] | None = None) -> None:
        self.group_k = group_k or {}

    def fit(self, x_rows: list[dict[str, float]], y: Any = None) -> "GroupedDictVectorizerSelector":
        self.group_order_ = [group for group in ("static", "behavior", "consistency", "other") if group in self.group_k]
        for group in ("static", "behavior", "consistency", "other"):
            if group not in self.group_order_:
                self.group_order_.append(group)

        self.vectorizers_: dict[str, DictVectorizer] = {}
        self.selectors_: dict[str, CappedSelectKBest] = {}
        self.group_feature_counts_: dict[str, int] = {}
        self.group_selected_feature_counts_: dict[str, int] = {}
        self.vocabulary_: dict[str, int] = {}
        raw_offset = 0

        grouped_rows = self._split_rows(x_rows)
        for group in self.group_order_:
            rows = grouped_rows.get(group, [])
            if not rows or not any(row for row in rows):
                self.group_feature_counts_[group] = 0
                self.group_selected_feature_counts_[group] = 0
                continue

            vectorizer = DictVectorizer(sparse=True)
            matrix = vectorizer.fit_transform(rows)
            feature_count = int(matrix.shape[1])
            selector = CappedSelectKBest(k=int(self.group_k.get(group, 0)))
            selector.fit(matrix, y)

            self.vectorizers_[group] = vectorizer
            self.selectors_[group] = selector
            self.group_feature_counts_[group] = feature_count
            self.group_selected_feature_counts_[group] = int(selector.selected_feature_count_)
            for name, index in vectorizer.vocabulary_.items():
                self.vocabulary_[name] = raw_offset + int(index)
            raw_offset += feature_count

        self.selected_feature_count_ = int(sum(self.group_selected_feature_counts_.values()))
        return self

    def transform(self, x_rows: list[dict[str, float]]) -> Any:
        grouped_rows = self._split_rows(x_rows)
        matrices = []
        for group in self.group_order_:
            vectorizer = self.vectorizers_.get(group)
            if vectorizer is None:
                continue
            matrix = vectorizer.transform(grouped_rows.get(group, [{} for _ in x_rows]))
            selector = self.selectors_.get(group)
            if selector is not None:
                matrix = selector.transform(matrix)
            matrices.append(matrix)
        if not matrices:
            return sparse.csr_matrix((len(x_rows), 0))
        return sparse.hstack(matrices, format="csr")

    def _split_rows(self, x_rows: list[dict[str, float]]) -> dict[str, list[dict[str, float]]]:
        grouped_rows: dict[str, list[dict[str, float]]] = {
            "static": [],
            "behavior": [],
            "consistency": [],
            "other": [],
        }
        for row in x_rows:
            grouped = {"static": {}, "behavior": {}, "consistency": {}, "other": {}}
            for key, value in row.items():
                grouped[feature_group(key)][key] = value
            for group, group_row in grouped.items():
                grouped_rows[group].append(group_row)
        return grouped_rows


def feature_group(feature_name: str) -> str:
    if feature_name.startswith("static::") or feature_name.startswith("count::"):
        return "static"
    if feature_name.startswith("ablation::behavior") or feature_name.startswith("behavior_v2::"):
        return "behavior"
    if (
        feature_name.startswith("ablation::consistency")
        or feature_name.startswith("ablation::support")
        or feature_name.startswith("ablation::adaptive")
        or feature_name.startswith("ablation::evidence")
        or feature_name.startswith("ablation::view")
        or feature_name.startswith("consistency_v2::")
    ):
        return "consistency"
    return "other"


def _evaluate_classifier(
    classifier_name: str,
    model: Pipeline,
    model_path: Path,
    y_test: list[int],
    predictions: Any,
    scores: list[float],
    test_sample_ids: list[str],
    fit_seconds: float,
    predict_seconds: float,
) -> dict[str, Any]:
    feature_meta = feature_selection_metadata(model)
    metrics: dict[str, Any] = {
        "classifier": classifier_name,
        "display_name": CLASSIFIER_DISPLAY_NAMES[classifier_name],
        "status": "ok",
        **feature_meta,
        "model_path": str(model_path),
        "model_size_bytes": int(model_path.stat().st_size),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "avg_predict_seconds_per_sample": float(predict_seconds / max(1, len(y_test))),
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


def feature_selection_metadata(model: Pipeline) -> dict[str, Any]:
    vectorizer = model.named_steps["vectorizer"]
    selector = model.named_steps.get("select")
    raw_count = int(len(vectorizer.vocabulary_))
    selected_count = int(
        getattr(
            selector,
            "selected_feature_count_",
            getattr(vectorizer, "selected_feature_count_", raw_count),
        )
    )
    metadata: dict[str, Any] = {
        "feature_count": raw_count,
        "selected_feature_count": selected_count,
    }
    if hasattr(vectorizer, "group_feature_counts_"):
        metadata["group_feature_counts"] = dict(vectorizer.group_feature_counts_)
        metadata["group_selected_feature_counts"] = dict(vectorizer.group_selected_feature_counts_)
    return metadata


def _decision_scores(model: Pipeline, x_rows: list[dict[str, float]]) -> list[float]:
    classifier = model.named_steps["classifier"]
    transformed = model[:-1].transform(x_rows)
    if hasattr(classifier, "predict_proba"):
        return [float(row[1]) for row in classifier.predict_proba(transformed)]
    if hasattr(classifier, "decision_function"):
        raw_scores = classifier.decision_function(transformed)
        return [float(item) for item in raw_scores]
    return [float(item) for item in classifier.predict(transformed)]


def _best_classifier(metrics_rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [
        row
        for row in metrics_rows
        if row.get("status") == "ok" and row.get(key) is not None
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda row: (float(row[key]), float(row.get("f1", 0.0))))
    return {
        "classifier": best["classifier"],
        "display_name": best["display_name"],
        key: best[key],
        "f1": best.get("f1"),
    }


def _to_dense(matrix: Any) -> Any:
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix
