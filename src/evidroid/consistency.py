from __future__ import annotations

from collections import Counter
from typing import Any

from .config import VIEW_WEIGHTS


def support_by_view(evidence_ids: list[str], evidence_index: dict[str, dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for evidence_id in evidence_ids:
        item = evidence_index.get(evidence_id)
        if item:
            counter[item["view"]] += 1
    return dict(counter)


def consistency_score(
    evidence_ids: list[str],
    evidence_index: dict[str, dict[str, Any]],
    view_weights: dict[str, float] | None = None,
) -> float:
    weights = view_weights or VIEW_WEIGHTS
    views = {evidence_index[item]["view"] for item in evidence_ids if item in evidence_index}
    if not views:
        return 0.0
    max_weight = sum(weights.values())
    view_score = sum(weights.get(view, 0.0) for view in views) / max_weight
    support_score = min(1.0, len(set(evidence_ids)) / 6.0)
    return round((0.7 * view_score) + (0.3 * support_score), 4)
