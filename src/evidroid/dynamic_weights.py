from __future__ import annotations

from collections import defaultdict
from typing import Any

from evidroid.config import VIEW_WEIGHTS


VIEW_ORDER = ("permission", "api", "component", "string")


def fixed_weight_spec(base_weights: dict[str, float] | None = None) -> dict[str, Any]:
    base = normalize_weights(base_weights or VIEW_WEIGHTS)
    return {
        "mode": "fixed",
        "alpha": 0.0,
        "base_weights": base,
        "global_weights": base,
        "behavior_weights": {},
        "score_method": "fixed",
    }


def learn_view_weight_spec(
    behavior_docs: list[dict[str, Any]],
    labels: list[int],
    mode: str = "behavior",
    base_weights: dict[str, float] | None = None,
    alpha: float = 0.5,
    min_label_samples: int = 5,
    score_method: str = "chi2",
    shrinkage_samples: float = 25.0,
) -> dict[str, Any]:
    if mode not in {"global", "behavior"}:
        raise ValueError("mode must be one of: global, behavior")
    if score_method not in {"chi2", "malware_lift", "hybrid"}:
        raise ValueError("score_method must be one of: chi2, malware_lift, hybrid")
    if len(behavior_docs) != len(labels):
        raise ValueError("behavior_docs and labels must have the same length.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    base = normalize_weights(base_weights or VIEW_WEIGHTS)
    global_scores = _global_view_scores(behavior_docs, labels, score_method=score_method)
    learned_global = _scores_to_weights(global_scores, fallback=base)
    final_global = _smooth_weights(base, learned_global, alpha)

    behavior_weights: dict[str, dict[str, float]] = {}
    behavior_scores: dict[str, dict[str, float]] = {}
    label_sample_counts = _label_sample_counts(behavior_docs)

    if mode == "behavior":
        for behavior_label in sorted(label_sample_counts):
            if label_sample_counts[behavior_label] < min_label_samples:
                continue
            scores = _behavior_view_scores(
                behavior_docs,
                labels,
                behavior_label,
                score_method=score_method,
            )
            learned = _scores_to_weights(scores, fallback=final_global)
            behavior_alpha = _behavior_alpha(
                alpha=alpha,
                sample_count=label_sample_counts[behavior_label],
                shrinkage_samples=shrinkage_samples,
            )
            behavior_scores[behavior_label] = scores
            behavior_weights[behavior_label] = _smooth_weights(final_global, learned, behavior_alpha)

    return {
        "mode": mode,
        "alpha": float(alpha),
        "min_label_samples": int(min_label_samples),
        "shrinkage_samples": float(shrinkage_samples),
        "score_method": score_method,
        "base_weights": base,
        "global_scores": global_scores,
        "global_weights": final_global,
        "behavior_scores": behavior_scores,
        "behavior_weights": behavior_weights,
        "label_sample_counts": label_sample_counts,
    }


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    normalized = {view: max(0.0, float(weights.get(view, 0.0))) for view in VIEW_ORDER}
    total = sum(normalized.values())
    if total <= 0:
        even = 1.0 / len(VIEW_ORDER)
        return {view: even for view in VIEW_ORDER}
    return {view: normalized[view] / total for view in VIEW_ORDER}


def weights_for_behavior(weight_spec: dict[str, Any] | None, behavior_label: str | None = None) -> dict[str, float]:
    if not weight_spec:
        return normalize_weights(VIEW_WEIGHTS)
    mode = str(weight_spec.get("mode", "fixed"))
    if mode == "behavior" and behavior_label:
        behavior_weights = weight_spec.get("behavior_weights", {})
        if isinstance(behavior_weights, dict) and behavior_label in behavior_weights:
            return normalize_weights(behavior_weights[behavior_label])
    global_weights = weight_spec.get("global_weights") or weight_spec.get("base_weights") or VIEW_WEIGHTS
    return normalize_weights(global_weights)


def behavior_consistency_score(
    behavior: dict[str, Any],
    weight_spec: dict[str, Any] | None = None,
) -> float:
    if not weight_spec:
        return float(behavior.get("consistency_score", 0.0))

    support_by_view = {
        view: int(count)
        for view, count in behavior.get("support_by_view", {}).items()
        if int(count) > 0
    }
    if not support_by_view:
        return float(behavior.get("consistency_score", 0.0))

    weights = weights_for_behavior(weight_spec, behavior.get("label"))
    weight_total = sum(weights.values()) or 1.0
    view_score = sum(weights.get(view, 0.0) for view in support_by_view) / weight_total
    evidence_ids = behavior.get("evidence_ids", [])
    support_score = min(1.0, len(set(evidence_ids)) / 6.0)
    return round((0.7 * view_score) + (0.3 * support_score), 4)


def _global_view_scores(
    behavior_docs: list[dict[str, Any]],
    labels: list[int],
    score_method: str = "chi2",
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for view in VIEW_ORDER:
        values = [
            _doc_has_view_support(behavior_doc, view)
            for behavior_doc in behavior_docs
        ]
        scores[view] = _view_score(values, labels, method=score_method)
    return scores


def _behavior_view_scores(
    behavior_docs: list[dict[str, Any]],
    labels: list[int],
    behavior_label: str,
    score_method: str = "chi2",
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for view in VIEW_ORDER:
        values = [
            _doc_has_behavior_view_support(behavior_doc, behavior_label, view)
            for behavior_doc in behavior_docs
        ]
        scores[view] = _view_score(values, labels, method=score_method)
    return scores


def _label_sample_counts(behavior_docs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for behavior_doc in behavior_docs:
        labels = {
            item.get("label")
            for item in behavior_doc.get("behaviors", [])
            if item.get("label")
        }
        for label in labels:
            counts[str(label)] += 1
    return dict(counts)


def _doc_has_view_support(behavior_doc: dict[str, Any], view: str) -> int:
    for behavior in behavior_doc.get("behaviors", []):
        if int(behavior.get("support_by_view", {}).get(view, 0)) > 0:
            return 1
    return 0


def _doc_has_behavior_view_support(
    behavior_doc: dict[str, Any],
    behavior_label: str,
    view: str,
) -> int:
    for behavior in behavior_doc.get("behaviors", []):
        if behavior.get("label") != behavior_label:
            continue
        if int(behavior.get("support_by_view", {}).get(view, 0)) > 0:
            return 1
    return 0


def _binary_chi2(values: list[int], labels: list[int]) -> float:
    if not values or len(values) != len(labels):
        return 0.0
    a = sum(1 for value, label in zip(values, labels) if value and label == 1)
    b = sum(1 for value, label in zip(values, labels) if value and label == 0)
    c = sum(1 for value, label in zip(values, labels) if not value and label == 1)
    d = sum(1 for value, label in zip(values, labels) if not value and label == 0)
    n = a + b + c + d
    denominator = (a + b) * (c + d) * (a + c) * (b + d)
    if n == 0 or denominator == 0:
        return 0.0
    return float(n * ((a * d) - (b * c)) ** 2 / denominator)


def _view_score(values: list[int], labels: list[int], method: str) -> float:
    chi2 = _binary_chi2(values, labels)
    lift = _malware_lift(values, labels)
    if method == "chi2":
        return chi2
    if method == "malware_lift":
        return lift
    if method == "hybrid":
        return chi2 * lift
    raise ValueError(f"Unknown score method: {method}")


def _malware_lift(values: list[int], labels: list[int]) -> float:
    malware_total = sum(1 for label in labels if label == 1)
    benign_total = sum(1 for label in labels if label == 0)
    if malware_total == 0 or benign_total == 0:
        return 0.0
    malware_hits = sum(1 for value, label in zip(values, labels) if value and label == 1)
    benign_hits = sum(1 for value, label in zip(values, labels) if value and label == 0)
    malware_rate = malware_hits / malware_total
    benign_rate = benign_hits / benign_total
    return max(0.0, malware_rate - benign_rate)


def _scores_to_weights(
    scores: dict[str, float],
    fallback: dict[str, float],
) -> dict[str, float]:
    positive_scores = {view: max(0.0, float(scores.get(view, 0.0))) for view in VIEW_ORDER}
    if sum(positive_scores.values()) <= 0:
        return normalize_weights(fallback)
    return normalize_weights(positive_scores)


def _smooth_weights(
    base: dict[str, float],
    learned: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    combined = {
        view: ((1.0 - alpha) * base.get(view, 0.0)) + (alpha * learned.get(view, 0.0))
        for view in VIEW_ORDER
    }
    return normalize_weights(combined)


def _behavior_alpha(alpha: float, sample_count: int, shrinkage_samples: float) -> float:
    if shrinkage_samples <= 0:
        return alpha
    shrink = sample_count / (sample_count + shrinkage_samples)
    return alpha * shrink
