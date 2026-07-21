from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class Evidence:
    id: str
    view: str
    value: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BehaviorFinding:
    label: str
    name: str
    description: str
    evidence_ids: list[str]
    support_by_view: dict[str, int]
    consistency_score: float
    analyzer: str = "rules"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_evidence(evidence_doc: dict[str, Any]) -> Iterable[dict[str, Any]]:
    return evidence_doc.get("evidence", [])


def evidence_by_id(evidence_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in iter_evidence(evidence_doc)}


def group_evidence_by_view(evidence_doc: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in iter_evidence(evidence_doc):
        grouped.setdefault(item["view"], []).append(item)
    return grouped
