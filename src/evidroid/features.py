from __future__ import annotations

import re
from statistics import mean
from typing import Any

from evidroid.dynamic_weights import behavior_consistency_score
from evidroid.schemas import iter_evidence

TOKEN_RE = re.compile(r"[^a-zA-Z0-9_.$:/;-]+")
API_RE = re.compile(r"^(?P<class>[^;]+;?)->(?P<method>[^\s(]+)")
VIEW_ORDER = ("permission", "api", "component", "string")


def build_feature_dict(
    evidence_doc: dict[str, Any],
    behavior_doc: dict[str, Any] | None = None,
    mode: str = "fusion",
) -> dict[str, float]:
    if mode not in {"static", "behavior", "fusion"}:
        raise ValueError(f"Unknown feature mode: {mode}")

    features: dict[str, float] = {}
    if mode in {"static", "fusion"}:
        _add_static_features(features, evidence_doc)
    if mode in {"behavior", "fusion"}:
        _add_behavior_features(features, behavior_doc or {})
    return features


def build_ablation_feature_dict(
    evidence_doc: dict[str, Any],
    behavior_doc: dict[str, Any] | None = None,
    use_behavior_semantics: bool = False,
    use_consistency: bool = False,
    min_consistency: float = 0.0,
    min_support_views: int = 1,
    top_k_behaviors: int | None = None,
    static_profile: str = "basic",
    view_weight_spec: dict[str, Any] | None = None,
    feature_version: str = "v1",
) -> dict[str, float]:
    """Build features for the A0-A3 ablation study.

    A0: static only.
    A1: static + behavior semantic labels.
    A2: static + dynamic consistency statistics, without independent behavior
    semantic features.
    A3: static + behavior semantic labels + dynamic consistency features.
    """

    if feature_version not in {"v1", "v2"}:
        raise ValueError(f"Unknown feature version: {feature_version}")

    features: dict[str, float] = {}
    _add_static_features(features, evidence_doc, profile=static_profile)

    behavior_doc = filter_behavior_doc(
        behavior_doc or {},
        min_consistency=min_consistency,
        min_support_views=min_support_views,
        top_k_behaviors=top_k_behaviors,
    )
    if use_behavior_semantics:
        _add_behavior_semantic_features(
            features,
            behavior_doc,
            evidence_doc=evidence_doc,
            feature_version=feature_version,
        )
    if use_consistency:
        _add_consistency_features(
            features,
            behavior_doc,
            include_label_specific=use_behavior_semantics,
            view_weight_spec=view_weight_spec,
            feature_version=feature_version,
        )
    return features


def build_evidroid_feature_dict(
    evidence_doc: dict[str, Any],
    behavior_doc: dict[str, Any] | None = None,
    min_consistency: float = 0.0,
    min_support_views: int = 1,
    top_k_behaviors: int | None = None,
    static_profile: str = "basic",
    view_weight_spec: dict[str, Any] | None = None,
    feature_version: str = "v1",
) -> dict[str, float]:
    return build_ablation_feature_dict(
        evidence_doc,
        behavior_doc,
        use_behavior_semantics=True,
        use_consistency=True,
        min_consistency=min_consistency,
        min_support_views=min_support_views,
        top_k_behaviors=top_k_behaviors,
        static_profile=static_profile,
        view_weight_spec=view_weight_spec,
        feature_version=feature_version,
    )


def build_ablation_feature_parts(
    evidence_doc: dict[str, Any],
    behavior_doc: dict[str, Any],
    min_consistency: float = 0.0,
    min_support_views: int = 1,
    top_k_behaviors: int | None = None,
    static_profile: str = "basic",
    view_weight_spec: dict[str, Any] | None = None,
    feature_version: str = "v1",
) -> dict[str, dict[str, float]]:
    if feature_version not in {"v1", "v2"}:
        raise ValueError(f"Unknown feature version: {feature_version}")

    behavior_doc = filter_behavior_doc(
        behavior_doc,
        min_consistency=min_consistency,
        min_support_views=min_support_views,
        top_k_behaviors=top_k_behaviors,
    )
    static_features: dict[str, float] = {}
    behavior_features: dict[str, float] = {}
    consistency_features: dict[str, float] = {}
    behavior_consistency_features: dict[str, float] = {}

    _add_static_features(static_features, evidence_doc, profile=static_profile)
    _add_behavior_semantic_features(
        behavior_features,
        behavior_doc,
        evidence_doc=evidence_doc,
        feature_version=feature_version,
    )
    _add_consistency_features(
        consistency_features,
        behavior_doc,
        include_label_specific=False,
        view_weight_spec=view_weight_spec,
        feature_version=feature_version,
    )
    _add_consistency_features(
        behavior_consistency_features,
        behavior_doc,
        include_label_specific=True,
        view_weight_spec=view_weight_spec,
        feature_version=feature_version,
    )
    return {
        "static": static_features,
        "behavior": behavior_features,
        "consistency": consistency_features,
        "behavior_consistency": behavior_consistency_features,
    }


def filter_behavior_doc(
    behavior_doc: dict[str, Any],
    min_consistency: float = 0.0,
    min_support_views: int = 1,
    top_k_behaviors: int | None = None,
) -> dict[str, Any]:
    behaviors = []
    for item in behavior_doc.get("behaviors", []):
        score = float(item.get("consistency_score", 0.0))
        if score < min_consistency:
            continue
        if len(item.get("support_by_view", {})) < max(1, min_support_views):
            continue
        behaviors.append(item)
    behaviors.sort(key=lambda item: float(item.get("consistency_score", 0.0)), reverse=True)
    if top_k_behaviors is not None and top_k_behaviors > 0:
        behaviors = behaviors[:top_k_behaviors]
    return {
        **behavior_doc,
        "behaviors": behaviors,
    }


def _add_static_features(features: dict[str, float], evidence_doc: dict[str, Any], profile: str = "basic") -> None:
    if profile not in {"basic", "drebin", "compact"}:
        raise ValueError(f"Unknown static profile: {profile}")

    view_counts: dict[str, int] = {}
    suspicious_string_count = 0
    for item in iter_evidence(evidence_doc):
        view = item["view"]
        view_counts[view] = view_counts.get(view, 0) + 1
        if view == "string" and _is_suspicious_string(str(item.get("value", ""))):
            suspicious_string_count += 1
        value = normalize_feature_value(view, item["value"])
        if profile == "basic":
            features[f"static::{view}::{value}"] = 1.0
        elif profile == "compact":
            _add_compact_static_feature(features, item)
        elif view == "permission":
            features[f"static::requested_permission::{value}"] = 1.0
        elif view == "api":
            features[f"static::restricted_api::{value}"] = 1.0
        elif view == "component":
            component_type = item.get("detail", {}).get("component_type", "component")
            features[f"static::app_component::{component_type}"] = 1.0
            features[f"static::app_component_name::{value}"] = 1.0
        elif view == "string" and _is_suspicious_string(item["value"]):
            features[f"static::suspicious_string::{value}"] = 1.0

    for view, count in view_counts.items():
        features[f"count::{view}"] = float(count)
    if profile in {"drebin", "compact"}:
        features["count::suspicious_string"] = float(suspicious_string_count)


def _add_behavior_features(features: dict[str, float], behavior_doc: dict[str, Any]) -> None:
    behaviors = behavior_doc.get("behaviors", [])
    scores: list[float] = []
    for item in behaviors:
        label = item["label"]
        score = float(item.get("consistency_score", 0.0))
        scores.append(score)
        features[f"behavior::{label}"] = 1.0
        features[f"consistency::{label}"] = score
        for view, count in item.get("support_by_view", {}).items():
            features[f"behavior_support::{label}::{view}"] = float(count)

    features["behavior_count"] = float(len(behaviors))
    features["consistency_max"] = max(scores) if scores else 0.0
    features["consistency_mean"] = mean(scores) if scores else 0.0


def _add_behavior_semantic_features(
    features: dict[str, float],
    behavior_doc: dict[str, Any],
    evidence_doc: dict[str, Any] | None = None,
    feature_version: str = "v1",
) -> None:
    behaviors = behavior_doc.get("behaviors", [])
    labels = {item["label"] for item in behaviors if item.get("label")}
    for label in labels:
        features[f"ablation::behavior::{label}"] = 1.0
    features["ablation::behavior_count"] = float(len(labels))

    if feature_version == "v2":
        _add_llm_doc_risk_features(features, behavior_doc)
        _add_behavior_semantic_features_v2(features, behaviors, evidence_doc or {})


def _add_consistency_features(
    features: dict[str, float],
    behavior_doc: dict[str, Any],
    include_label_specific: bool,
    view_weight_spec: dict[str, Any] | None = None,
    feature_version: str = "v1",
) -> None:
    behaviors = behavior_doc.get("behaviors", [])
    scores: list[float] = []
    evidence_counts: list[int] = []
    view_counts: list[int] = []
    support_totals: dict[str, int] = {}

    for item in behaviors:
        fixed_score = float(item.get("consistency_score", 0.0))
        dynamic_score = behavior_consistency_score(item, view_weight_spec)
        score = fixed_score if _augment_fixed_consistency(view_weight_spec) else dynamic_score
        scores.append(score)
        evidence_ids = item.get("evidence_ids", [])
        support_by_view = item.get("support_by_view", {})
        evidence_counts.append(len(evidence_ids))
        view_counts.append(len(support_by_view))
        for view, count in support_by_view.items():
            support_totals[view] = support_totals.get(view, 0) + int(count)

        if include_label_specific and item.get("label"):
            label = item["label"]
            features[f"ablation::consistency::{label}"] = score
            for view, count in support_by_view.items():
                features[f"ablation::support::{label}::{view}"] = float(count)
            if _augment_fixed_consistency(view_weight_spec):
                features[f"ablation::adaptive_consistency::{label}"] = dynamic_score
                delta = dynamic_score - fixed_score
                features[f"ablation::adaptive_delta_pos::{label}"] = max(0.0, delta)
                features[f"ablation::adaptive_delta_neg::{label}"] = max(0.0, -delta)

    features["ablation::consistency_count"] = float(len(scores))
    features["ablation::consistency_max"] = max(scores) if scores else 0.0
    features["ablation::consistency_mean"] = mean(scores) if scores else 0.0
    features["ablation::consistency_sum"] = sum(scores)
    features["ablation::evidence_count_max"] = float(max(evidence_counts)) if evidence_counts else 0.0
    features["ablation::evidence_count_mean"] = mean(evidence_counts) if evidence_counts else 0.0
    features["ablation::view_count_max"] = float(max(view_counts)) if view_counts else 0.0
    features["ablation::view_count_mean"] = mean(view_counts) if view_counts else 0.0
    for view, count in support_totals.items():
        features[f"ablation::support_total::{view}"] = float(count)

    if _augment_fixed_consistency(view_weight_spec):
        adaptive_scores = [behavior_consistency_score(item, view_weight_spec) for item in behaviors]
        features["ablation::adaptive_consistency_max"] = max(adaptive_scores) if adaptive_scores else 0.0
        features["ablation::adaptive_consistency_mean"] = mean(adaptive_scores) if adaptive_scores else 0.0
        features["ablation::adaptive_consistency_sum"] = sum(adaptive_scores)

    if feature_version == "v2":
        _add_consistency_features_v2(
            features,
            behaviors,
            include_label_specific=include_label_specific,
            view_weight_spec=view_weight_spec,
        )


def normalize_feature_value(view: str, value: str) -> str:
    value = value.strip()
    if view == "permission":
        value = value.split(".")[-1]
    elif view == "api":
        value = value.replace("Landroid/", "android/").replace("Ljava/", "java/")
    elif view == "string":
        value = value.lower()
    value = TOKEN_RE.sub("_", value)
    return value[:140]


def _add_compact_static_feature(features: dict[str, float], item: dict[str, Any]) -> None:
    view = item["view"]
    raw_value = str(item.get("value", ""))
    if view == "permission":
        permission = normalize_feature_value("permission", raw_value)
        features[f"static::permission::{permission}"] = 1.0
        features[f"static::permission_group::{_permission_group(permission)}"] = 1.0
    elif view == "api":
        api_meta = _api_metadata(raw_value)
        for key, value in api_meta.items():
            features[f"static::{key}::{value}"] = 1.0
    elif view == "component":
        component_type = str(item.get("detail", {}).get("component_type", "component"))
        features[f"static::component_type::{component_type}"] = 1.0
        for token in _component_tokens(raw_value):
            features[f"static::component_token::{token}"] = 1.0
    elif view == "string":
        markers = _string_markers(raw_value)
        if markers:
            for marker in markers:
                features[f"static::string_marker::{marker}"] = 1.0
        else:
            features[f"static::string_shape::{_string_shape(raw_value)}"] = 1.0


def _add_behavior_semantic_features_v2(
    features: dict[str, float],
    behaviors: list[dict[str, Any]],
    evidence_doc: dict[str, Any],
) -> None:
    evidence_index = {item["id"]: item for item in iter_evidence(evidence_doc)}
    labels = sorted({str(item.get("label")) for item in behaviors if item.get("label")})
    sources_by_label: dict[str, set[str]] = {}
    source_counts: dict[str, int] = {}
    for item in behaviors:
        label = str(item.get("label", ""))
        source = _behavior_source(item)
        if not label or not source:
            continue
        sources_by_label.setdefault(label, set()).add(source)
        source_counts[source] = source_counts.get(source, 0) + 1

    for label in labels:
        features[f"behavior_v2::{label}::present"] = 1.0
        sources = sources_by_label.get(label, set())
        if sources:
            features[f"behavior_v2::{label}::source_count"] = float(len(sources))
            features[f"behavior_v2::{label}::source_count_bucket::{_count_bucket(len(sources))}"] = 1.0
        if len(sources) >= 2:
            features[f"behavior_v2::{label}::llm_prompt_agreement"] = 1.0

    for first, second in _pairwise(labels):
        features[f"behavior_v2::cooccur::{first}+{second}"] = 1.0

    features["behavior_v2::label_count"] = float(len(labels))
    features[f"behavior_v2::label_count_bucket::{_count_bucket(len(labels))}"] = 1.0
    features["behavior_v2::finding_count"] = float(len(behaviors))
    features[f"behavior_v2::finding_count_bucket::{_count_bucket(len(behaviors))}"] = 1.0
    agreement_count = sum(1 for sources in sources_by_label.values() if len(sources) >= 2)
    if sources_by_label:
        features["behavior_v2::llm_prompt_agreement_count"] = float(agreement_count)
        features["behavior_v2::llm_prompt_agreement_ratio"] = agreement_count / max(1.0, float(len(sources_by_label)))
        features[f"behavior_v2::llm_prompt_agreement_bucket::{_count_bucket(agreement_count)}"] = 1.0
    for source, count in source_counts.items():
        features[f"behavior_v2::source::{source}::finding_count"] = float(count)
        features[f"behavior_v2::source::{source}::finding_bucket::{_count_bucket(count)}"] = 1.0

    for item in behaviors:
        label = str(item.get("label", "unknown"))
        score = float(item.get("consistency_score", 0.0))
        source = _behavior_source(item)
        malware_relevance = _optional_unit_feature(item.get("malware_relevance"))
        confidence = _optional_unit_feature(item.get("confidence"))
        risk_level = _normalize_category(str(item.get("risk_level", "") or ""))
        evidence_ids = list(item.get("evidence_ids", []))
        support_by_view = {
            str(view): int(count)
            for view, count in item.get("support_by_view", {}).items()
            if int(count) > 0
        }
        views = sorted(support_by_view, key=_view_sort_key)
        view_mask = _view_mask(views)

        _set_max(features, f"behavior_v2::{label}::score_max", score)
        _set_max(features, f"behavior_v2::{label}::evidence_count_max", float(len(set(evidence_ids))))
        _set_max(features, f"behavior_v2::{label}::support_view_count_max", float(len(views)))
        features[f"behavior_v2::{label}::score_bucket::{_score_bucket(score)}"] = 1.0
        features[f"behavior_v2::{label}::evidence_bucket::{_count_bucket(len(set(evidence_ids)))}"] = 1.0
        features[f"behavior_v2::{label}::view_mask::{view_mask}"] = 1.0
        if source:
            features[f"behavior_v2::{label}::source::{source}"] = 1.0
            features[f"behavior_v2::source_label::{source}::{label}"] = 1.0
        if malware_relevance is not None:
            _set_max(features, f"behavior_v2::{label}::malware_relevance_max", malware_relevance)
            features[f"behavior_v2::{label}::malware_relevance_bucket::{_score_bucket(malware_relevance)}"] = 1.0
        if confidence is not None:
            _set_max(features, f"behavior_v2::{label}::llm_confidence_max", confidence)
            features[f"behavior_v2::{label}::llm_confidence_bucket::{_score_bucket(confidence)}"] = 1.0
        if risk_level:
            features[f"behavior_v2::{label}::risk_level::{risk_level}"] = 1.0

        for view, count in support_by_view.items():
            features[f"behavior_v2::{label}::view::{view}"] = 1.0
            _set_max(features, f"behavior_v2::{label}::view_count::{view}", float(count))

        for evidence_id in evidence_ids:
            evidence_item = evidence_index.get(evidence_id)
            if not evidence_item:
                continue
            for token in _behavior_evidence_tokens(evidence_item):
                features[f"behavior_v2::{label}::{token}"] = 1.0


def _add_consistency_features_v2(
    features: dict[str, float],
    behaviors: list[dict[str, Any]],
    include_label_specific: bool,
    view_weight_spec: dict[str, Any] | None = None,
) -> None:
    score_values: list[float] = []
    support_view_counts: list[int] = []
    evidence_counts: list[int] = []
    mask_counts: dict[str, int] = {}
    pair_counts: dict[str, int] = {}
    source_scores: dict[str, list[float]] = {}
    source_counts: dict[str, int] = {}
    label_source_scores: dict[str, dict[str, list[float]]] = {}

    for item in behaviors:
        score = behavior_consistency_score(item, view_weight_spec)
        score_values.append(score)
        source = _behavior_source(item)
        label = str(item.get("label", ""))
        malware_relevance = _optional_unit_feature(item.get("malware_relevance"))
        confidence = _optional_unit_feature(item.get("confidence"))
        if source:
            source_scores.setdefault(source, []).append(score)
            source_counts[source] = source_counts.get(source, 0) + 1
            if label:
                label_source_scores.setdefault(label, {}).setdefault(source, []).append(score)
        evidence_count = len(set(item.get("evidence_ids", [])))
        evidence_counts.append(evidence_count)
        views = sorted(
            [str(view) for view, count in item.get("support_by_view", {}).items() if int(count) > 0],
            key=_view_sort_key,
        )
        support_view_counts.append(len(views))
        mask = _view_mask(views)
        mask_counts[mask] = mask_counts.get(mask, 0) + 1
        for first, second in _pairwise(views):
            key = f"{first}+{second}"
            pair_counts[key] = pair_counts.get(key, 0) + 1

        features[f"consistency_v2::score_bucket::{_score_bucket(score)}"] = (
            features.get(f"consistency_v2::score_bucket::{_score_bucket(score)}", 0.0) + 1.0
        )
        features[f"consistency_v2::support_view_bucket::{_count_bucket(len(views))}"] = (
            features.get(f"consistency_v2::support_view_bucket::{_count_bucket(len(views))}", 0.0) + 1.0
        )
        features[f"consistency_v2::evidence_bucket::{_count_bucket(evidence_count)}"] = (
            features.get(f"consistency_v2::evidence_bucket::{_count_bucket(evidence_count)}", 0.0) + 1.0
        )

        if include_label_specific and item.get("label"):
            label = str(item["label"])
            _set_max(features, f"consistency_v2::{label}::score_max", score)
            _set_max(features, f"consistency_v2::{label}::support_view_count_max", float(len(views)))
            _set_max(features, f"consistency_v2::{label}::evidence_count_max", float(evidence_count))
            features[f"consistency_v2::{label}::score_bucket::{_score_bucket(score)}"] = 1.0
            features[f"consistency_v2::{label}::view_mask::{mask}"] = 1.0
            if source:
                _set_max(features, f"consistency_v2::{label}::source::{source}::score_max", score)
            if malware_relevance is not None:
                _set_max(features, f"consistency_v2::{label}::malware_relevance_max", malware_relevance)
            if confidence is not None:
                _set_max(features, f"consistency_v2::{label}::llm_confidence_max", confidence)
            for first, second in _pairwise(views):
                features[f"consistency_v2::{label}::view_pair::{first}+{second}"] = 1.0

    features["consistency_v2::finding_count"] = float(len(behaviors))
    features["consistency_v2::score_max"] = max(score_values) if score_values else 0.0
    features["consistency_v2::score_mean"] = mean(score_values) if score_values else 0.0
    features["consistency_v2::support_view_count_max"] = float(max(support_view_counts)) if support_view_counts else 0.0
    features["consistency_v2::support_view_count_mean"] = mean(support_view_counts) if support_view_counts else 0.0
    features["consistency_v2::evidence_count_max"] = float(max(evidence_counts)) if evidence_counts else 0.0
    features["consistency_v2::evidence_count_mean"] = mean(evidence_counts) if evidence_counts else 0.0

    for mask, count in mask_counts.items():
        features[f"consistency_v2::view_mask::{mask}"] = float(count)
    for pair, count in pair_counts.items():
        features[f"consistency_v2::view_pair::{pair}"] = float(count)
    for source, values in source_scores.items():
        features[f"consistency_v2::source::{source}::count"] = float(source_counts[source])
        features[f"consistency_v2::source::{source}::score_max"] = max(values)
        features[f"consistency_v2::source::{source}::score_mean"] = mean(values)
    for label, scores_by_source in label_source_scores.items():
        if len(scores_by_source) < 2:
            continue
        source_means = [mean(values) for values in scores_by_source.values()]
        features[f"consistency_v2::{label}::llm_prompt_agreement"] = 1.0
        features[f"consistency_v2::{label}::source_score_mean"] = mean(source_means)
        features[f"consistency_v2::{label}::source_score_range"] = max(source_means) - min(source_means)


def _behavior_source(item: dict[str, Any]) -> str:
    source = str(item.get("llm_prompt", "") or item.get("source", "")).strip()
    if source:
        return normalize_feature_value("source", source).lower()
    analyzer = str(item.get("analyzer", "")).lower()
    source_markers = {
        "default": ("rawllm", "default_prompt", "prompt_default", "llm_raw"),
        "malware_focused": ("focusedllm", "malware_focused", "focused_prompt"),
    }
    for name, markers in source_markers.items():
        if any(marker in analyzer for marker in markers):
            return name
    return ""


def _add_llm_doc_risk_features(features: dict[str, float], behavior_doc: dict[str, Any]) -> None:
    llm_risk = behavior_doc.get("llm_risk", {})
    if not isinstance(llm_risk, dict):
        return
    score = _optional_unit_feature(llm_risk.get("apk_risk_score"))
    if score is not None:
        features["behavior_v2::llm_doc_risk_score"] = score
        features[f"behavior_v2::llm_doc_risk_bucket::{_score_bucket(score)}"] = 1.0
    risk_level = _normalize_category(str(llm_risk.get("risk_level", "") or ""))
    if risk_level:
        features[f"behavior_v2::llm_doc_risk_level::{risk_level}"] = 1.0


def _optional_unit_feature(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, parsed))


def _normalize_category(value: str) -> str:
    value = value.strip().lower()
    value = TOKEN_RE.sub("_", value)
    return value[:80]


def _api_metadata(api_value: str) -> dict[str, str]:
    normalized = api_value.replace("Landroid/", "android/").replace("Ljava/", "java/")
    match = API_RE.search(normalized)
    class_name = match.group("class") if match else normalized.split("->", 1)[0]
    method_name = match.group("method") if match else "unknown"
    class_name = class_name.strip("L;").replace("/", ".").replace("$", ".")
    parts = [part for part in class_name.split(".") if part]
    package2 = ".".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "unknown")
    package3 = ".".join(parts[:3]) if len(parts) >= 3 else package2
    class_token = normalize_feature_value("api", parts[-1] if parts else "unknown").lower()
    method_token = normalize_feature_value("api", method_name).lower()
    family = _api_family(parts, method_token)
    return {
        "api_package2": package2.lower(),
        "api_package3": package3.lower(),
        "api_class": class_token,
        "api_method": method_token,
        "api_family": family,
    }


def _api_family(parts: list[str], method_name: str) -> str:
    text = ".".join(parts).lower() + "." + method_name.lower()
    families = {
        "sms_call": ("sms", "sendtextmessage", "action_call", "action_dial"),
        "telephony": ("telephony", "getdeviceid", "getsubscriberid", "getline1number", "getsimserialnumber"),
        "location": ("location", "gps", "requestlocationupdates", "getlastknownlocation"),
        "network": ("http", "urlconnection", "socket", "okhttp", "webview", "loadurl"),
        "reflection": ("classloader", "dexclassloader", "loadclass", "forname", "invoke"),
        "crypto": ("cipher", "messagedigest", "secretkeyspec", "base64", "mac"),
        "file": ("fileinputstream", "fileoutputstream", "openfileoutput", "externalstorage"),
        "package": ("packageinstaller", "packagemanager", "installpackage"),
        "native": ("loadlibrary", "system.load"),
        "process": ("runtime", "processbuilder", "exec"),
    }
    for family, keywords in families.items():
        if any(keyword in text for keyword in keywords):
            return family
    if parts[:1] == ["android"]:
        return "android_other"
    if parts[:1] == ["java"] or parts[:1] == ["javax"]:
        return "java_other"
    return "other"


def _permission_group(permission: str) -> str:
    upper = permission.upper()
    groups = {
        "sms": ("SMS",),
        "phone": ("PHONE", "CALL", "READ_PHONE_STATE"),
        "contacts": ("CONTACTS", "ACCOUNTS"),
        "location": ("LOCATION",),
        "network": ("INTERNET", "NETWORK", "WIFI"),
        "storage": ("STORAGE", "EXTERNAL", "MEDIA"),
        "boot": ("BOOT",),
        "overlay_accessibility": ("ALERT_WINDOW", "ACCESSIBILITY"),
        "install": ("INSTALL", "PACKAGE"),
    }
    for group, keywords in groups.items():
        if any(keyword in upper for keyword in keywords):
            return group
    return "other"


def _component_tokens(value: str) -> list[str]:
    raw_name = value.split(":", 1)[-1]
    tokens = []
    for part in re.split(r"[^A-Za-z0-9]+", raw_name):
        if 3 <= len(part) <= 32:
            token = normalize_feature_value("component", part).lower()
            if token in {"activity", "service", "receiver", "provider", "main", "boot", "sms"}:
                tokens.append(token)
    return tokens[:4]


def _string_markers(value: str) -> list[str]:
    lower = value.lower()
    markers = []
    marker_checks = {
        "url_http": ("http://", "https://"),
        "content_uri": ("content://",),
        "apk": (".apk",),
        "dex": (".dex", "classes.dex"),
        "native_so": (".so", "lib/"),
        "shell_path": ("/system/bin", "/system/xbin", "/bin/sh", "chmod ", "mount -o"),
        "sms": ("sms", "content://sms"),
        "device_id": ("imei", "imsi", "android_id", "device_id", "subscriber"),
        "credential": ("token", "password", "passwd", "secret"),
        "location": ("latitude", "longitude", "gps", "location"),
        "network_word": ("upload", "download", "server", "host", "api/"),
        "crypto_word": ("aes", "rsa", "md5", "sha-1", "sha256", "base64"),
    }
    for marker, keywords in marker_checks.items():
        if any(keyword in lower for keyword in keywords):
            markers.append(marker)
    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", lower):
        markers.append("ip_address")
    return markers[:6]


def _string_shape(value: str) -> str:
    if value.startswith("/"):
        return "path"
    if value.isdigit():
        return "numeric"
    if any(char.isdigit() for char in value) and any(char.isalpha() for char in value):
        return "alpha_numeric"
    return "plain"


def _behavior_evidence_tokens(item: dict[str, Any]) -> list[str]:
    view = item.get("view")
    value = str(item.get("value", ""))
    if view == "permission":
        permission = normalize_feature_value("permission", value)
        return [
            f"perm::{permission}",
            f"perm_group::{_permission_group(permission)}",
            f"exact::permission::{permission}",
        ]
    if view == "api":
        meta = _api_metadata(value)
        exact_api = normalize_feature_value("api", value).lower()
        return [
            f"api_family::{meta['api_family']}",
            f"api_package2::{meta['api_package2']}",
            f"api_package3::{meta['api_package3']}",
            f"api_class::{meta['api_class']}",
            f"api_method::{meta['api_method']}",
            f"exact::api::{exact_api}",
        ]
    if view == "component":
        component_type = str(item.get("detail", {}).get("component_type", "component"))
        component = normalize_feature_value("component", value).lower()
        return [
            f"component_type::{component_type}",
            f"exact::component::{component_type}::{component}",
        ]
    if view == "string":
        tokens = [f"string_marker::{marker}" for marker in _string_markers(value)]
        if _is_suspicious_string(value):
            tokens.append(f"exact::string::{normalize_feature_value('string', value).lower()}")
        return tokens
    return []


def _score_bucket(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.55:
        return "medium"
    if score > 0:
        return "low"
    return "zero"


def _count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3_4"
    if count <= 8:
        return "5_8"
    return "9_plus"


def _view_mask(views: list[str]) -> str:
    return "+".join(views) if views else "none"


def _view_sort_key(view: str) -> tuple[int, str]:
    try:
        return (VIEW_ORDER.index(view), view)
    except ValueError:
        return (len(VIEW_ORDER), view)


def _pairwise(values: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for idx, first in enumerate(values):
        for second in values[idx + 1 :]:
            pairs.append((first, second))
    return pairs


def _set_max(features: dict[str, float], key: str, value: float) -> None:
    features[key] = max(float(features.get(key, 0.0)), float(value))


def _is_suspicious_string(value: str) -> bool:
    lower = value.lower()
    return (
        "http://" in lower
        or "https://" in lower
        or "content://" in lower
        or lower.startswith("/")
        or ".apk" in lower
        or ".dex" in lower
        or ".so" in lower
        or "sms" in lower
        or "imei" in lower
        or "imsi" in lower
        or "android_id" in lower
    )


def _augment_fixed_consistency(view_weight_spec: dict[str, Any] | None) -> bool:
    return bool(view_weight_spec and view_weight_spec.get("augment_fixed"))
