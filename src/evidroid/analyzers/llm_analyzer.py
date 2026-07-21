from __future__ import annotations

import json
import re
from typing import Any

from evidroid.config import BEHAVIOR_RULES, SUSPICIOUS_STRING_MARKERS
from evidroid.consistency import consistency_score, support_by_view
from evidroid.schemas import evidence_by_id, group_evidence_by_view

DEFAULT_VIEW_BUDGETS = {
    "permission": 80,
    "api": 80,
    "component": 80,
    "string": 80,
}

COMPACT_VIEW_BUDGETS = {
    "permission": 80,
    "api": 40,
    "component": 30,
    "string": 25,
}

ADAPTIVE_BUDGET_MODES = {"legacy", "compact", "adaptive"}

URL_RE = re.compile(r"https?://([^/\s?#]+)(/[^\s?#]*)?", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


class OpenAIBehaviorAnalyzer:
    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: str | None = None,
        base_url: str | None = "https://api.deepseek.com",
        max_evidence_per_view: int = 80,
        max_tokens: int | None = 4096,
        temperature: float | None = 0,
        thinking: dict[str, Any] | str | bool | None = None,
        reasoning_effort: str | None = None,
        provider: str = "deepseek",
        request_timeout: float | None = 120.0,
        prompt_mode: str = "default",
        evidence_budget_mode: str = "legacy",
        view_budgets: dict[str, int] | None = None,
        max_value_chars: int | None = None,
        compact_evidence: bool = False,
    ) -> None:
        if prompt_mode not in {"default", "malware_focused", "risk_focused"}:
            raise ValueError(f"Unsupported LLM prompt mode: {prompt_mode}")
        evidence_budget_mode = evidence_budget_mode.strip().lower()
        if evidence_budget_mode not in ADAPTIVE_BUDGET_MODES:
            raise ValueError(f"Unsupported evidence budget mode: {evidence_budget_mode}")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_evidence_per_view = max_evidence_per_view
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.provider = provider
        self.request_timeout = request_timeout
        self.prompt_mode = prompt_mode
        self.evidence_budget_mode = evidence_budget_mode
        self.view_budgets = _normalize_view_budgets(view_budgets)
        self.max_value_chars = max_value_chars
        self.compact_evidence = compact_evidence

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OpenAIBehaviorAnalyzer":
        return cls(
            provider=str(config.get("provider", "deepseek")),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            model=str(config.get("model", "deepseek-v4-flash")),
            max_evidence_per_view=int(config.get("max_evidence_per_view", 80)),
            max_tokens=config.get("max_tokens"),
            temperature=config.get("temperature"),
            thinking=config.get("thinking"),
            reasoning_effort=config.get("reasoning_effort"),
            request_timeout=_optional_float(config.get("request_timeout", 120.0)),
            prompt_mode=str(config.get("prompt_mode", "default")),
            evidence_budget_mode=str(config.get("evidence_budget_mode", "legacy")),
            view_budgets=config.get("view_budgets"),
            max_value_chars=_optional_int(config.get("max_value_chars")),
            compact_evidence=bool(config.get("compact_evidence", False)),
        )

    def analyze(self, evidence_doc: dict[str, Any]) -> dict[str, Any]:
        from openai import OpenAI

        if not self.api_key:
            raise ValueError(
                "DeepSeek API key is empty. Fill configs/deepseek.json llm.api_key "
                "or set DEEPSEEK_API_KEY."
            )
        client_kwargs: dict[str, Any] = {"api_key": self.api_key, "base_url": self.base_url}
        if self.request_timeout is not None:
            client_kwargs["timeout"] = float(self.request_timeout)
        client = OpenAI(**client_kwargs)
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(),
            },
            {
                "role": "user",
                "content": self._build_prompt(evidence_doc),
            },
        ]
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        if self.max_tokens is not None:
            request["max_tokens"] = int(self.max_tokens)
        thinking_type = _thinking_type(self.thinking)
        if thinking_type:
            request["extra_body"] = {"thinking": {"type": thinking_type}}
        if thinking_type == "enabled" and self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        if thinking_type != "enabled" and self.temperature is not None:
            request["temperature"] = float(self.temperature)

        response = client.chat.completions.create(**request)
        content = response.choices[0].message.content or "{}"
        payload = _loads_json_object(content)
        result = self._validate(evidence_doc, payload)
        usage = getattr(response, "usage", None)
        if usage is not None:
            result["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return result

    def _system_prompt(self) -> str:
        if self.prompt_mode == "risk_focused":
            return (
                "You are an Android malware static-analysis assistant. "
                "Estimate malware-discriminative risk only from provided evidence IDs; "
                "do not use labels, dataset priors, package reputation, or external knowledge. "
                "Return strict JSON (a json object) with top-level keys 'apk_risk_score', "
                "'risk_level', 'risk_rationale', and 'behaviors'."
            )
        return (
            "You are an Android malware static-analysis assistant. "
            "Do not classify the APK as benign or malicious. "
            "Infer possible security behaviors only from provided evidence IDs. "
            f"Use prompt mode: {self.prompt_mode}. "
            "Return strict JSON (a json object) with a top-level key 'behaviors'."
        )

    def _build_prompt(self, evidence_doc: dict[str, Any]) -> str:
        taxonomy = [
            {"label": rule["label"], "name": rule["name"], "description": rule["description"]}
            for rule in BEHAVIOR_RULES
        ]
        grouped = group_evidence_by_view(evidence_doc)
        compact: dict[str, list[dict[str, str] | list[str]]] = {}
        evidence_budget: dict[str, dict[str, int]] = {}
        for view, items in grouped.items():
            selected = self._select_evidence_items(view, items)
            compact[view] = [self._format_evidence_item(item) for item in selected]
            evidence_budget[view] = {"available": len(items), "sent": len(selected)}
        prompt: dict[str, Any] = {
            "task": "Infer possible security behaviors from evidence. Every behavior must cite existing evidence IDs.",
            "allowed_taxonomy": taxonomy,
            "output_requirement": "Return valid json only. Do not add markdown or explanation.",
            "output_schema": {
                "behaviors": [
                    {
                        "label": "one allowed label",
                        "name": "short behavior name",
                        "description": "one sentence",
                        "evidence_ids": ["PERM_0001", "API_0001"],
                    }
                ]
            },
            "evidence": compact,
        }
        if self.evidence_budget_mode != "legacy" or self.compact_evidence or self.max_value_chars:
            prompt["input_evidence_format"] = (
                "Each evidence item is [evidence_id, compact_value]. Evidence values may be truncated; "
                "only evidence IDs are authoritative."
                if self.compact_evidence
                else "Each evidence item has id and value fields. Only evidence IDs are authoritative."
            )
            prompt["evidence_budget"] = evidence_budget
        if self.prompt_mode == "malware_focused":
            prompt.update(
                {
                    "task": (
                        "Infer only security behaviors that are useful for Android malware detection. "
                        "Be conservative: return an empty behaviors list when evidence is generic."
                    ),
                    "selection_policy": [
                        "Return at most 6 strongest behaviors, sorted by evidence strength.",
                        "Do not infer from package names, UI words, generic file names, or URLs alone.",
                        "For network_communication, require INTERNET permission, network APIs, or concrete remote endpoint evidence.",
                        "For file_storage_access, require storage APIs, storage permissions, or concrete external-path operations.",
                        "For privacy_collection, require identifier/contact/location/account APIs or permissions, not generic words alone.",
                        "For crypto_or_obfuscation, require crypto/reflection/loader APIs or explicit obfuscation markers.",
                        "Prefer malware-discriminative allowed behavior families when evidenced: dynamic_code_loading, "
                        "native_code_loading, command_execution, sms_or_call_abuse, boot_persistence, overlay_or_phishing, "
                        "package_installation, privacy_collection paired with network_communication, and crypto_or_obfuscation.",
                        "Prefer behaviors supported by at least two evidence views; single-view string-only behaviors must be rare and explicit.",
                        "Each behavior should cite 2 to 20 strongest evidence IDs. Use existing evidence IDs only.",
                    ],
                }
            )
        elif self.prompt_mode == "risk_focused":
            risk_policy = [
                "apk_risk_score must be between 0 and 1: 0 means strongly benign-looking, 1 means strongly malware-like.",
                "risk_level must be one of low, medium, high, critical.",
                "Assign high risk only when multiple suspicious behaviors reinforce each other, such as "
                "dynamic code loading plus network endpoints, privacy collection plus upload paths, "
                "SMS/call abuse, boot persistence, overlay/phishing, package installation, command execution, "
                "native code loading, or crypto/obfuscation evidence.",
                "Common benign capabilities such as network communication, generic file storage, ads, analytics, "
                "and broad UI components should receive low malware_relevance unless paired with stronger evidence.",
                "For each behavior, provide malware_relevance and confidence as numbers between 0 and 1.",
                "Each behavior should cite 2 to 20 strongest evidence IDs. Use existing evidence IDs only.",
                "Return at most 8 strongest behaviors, sorted by malware_relevance then confidence.",
            ]
            if self.evidence_budget_mode != "legacy":
                risk_policy.extend(
                    [
                        "The evidence list is risk-prioritized and compressed; do not interpret the density of suspicious-looking "
                        "items as the prevalence of risk in the full APK.",
                        "For compressed evidence, apk_risk_score >= 0.70 requires concrete high-risk evidence such as "
                        "SMS/call abuse, command execution, package installation/dropper behavior, overlay/phishing, "
                        "boot persistence paired with another suspicious behavior, or dynamic/native code loading paired "
                        "with network, privacy collection, or persistence.",
                        "If the strongest findings are only common network, storage, crypto, location, ads, analytics, "
                        "or UI framework behavior, keep apk_risk_score below 0.55 even when those records were selected.",
                    ]
                )
            prompt.update(
                {
                    "task": (
                        "Infer malware-discriminative behavior semantics and a calibrated APK risk estimate. "
                        "The risk estimate must be based only on cited evidence and behavior combinations."
                    ),
                    "risk_policy": risk_policy,
                    "output_schema": {
                        "apk_risk_score": 0.0,
                        "risk_level": "low|medium|high|critical",
                        "risk_rationale": "one short sentence grounded in cited behaviors",
                        "behaviors": [
                            {
                                "label": "one allowed label",
                                "name": "short behavior name",
                                "description": "one sentence",
                                "evidence_ids": ["PERM_0001", "API_0001"],
                                "malware_relevance": 0.0,
                                "confidence": 0.0,
                                "risk_level": "low|medium|high|critical",
                            }
                        ],
                    },
                }
            )
        return json.dumps(prompt, ensure_ascii=False)

    def _select_evidence_items(self, view: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.evidence_budget_mode == "legacy":
            return items[: self.max_evidence_per_view]

        budget = self.view_budgets.get(view)
        if budget is None:
            budget = COMPACT_VIEW_BUDGETS.get(view, self.max_evidence_per_view)
        if budget <= 0:
            return []

        ranked: list[tuple[int, int, dict[str, Any]]] = []
        seen: set[str] = set()
        for index, item in enumerate(items):
            value = str(item.get("value", "") or "")
            signature = _normalized_signature(view, value)
            if signature in seen:
                continue
            seen.add(signature)
            ranked.append((self._evidence_priority(view, item), index, item))

        ranked.sort(key=lambda row: (-row[0], row[1]))
        return [item for _, _, item in ranked[:budget]]

    def _format_evidence_item(self, item: dict[str, Any]) -> dict[str, str] | list[str]:
        evidence_id = str(item.get("id", "") or "")
        value = self._format_evidence_value(str(item.get("value", "") or ""))
        if self.compact_evidence:
            return [evidence_id, value]
        return {"id": evidence_id, "value": value}

    def _format_evidence_value(self, value: str) -> str:
        text = SPACE_RE.sub(" ", value).strip()
        if self.evidence_budget_mode != "legacy":
            text = _strip_url_noise(text)
        if self.max_value_chars and self.max_value_chars > 0 and len(text) > self.max_value_chars:
            return text[: max(0, self.max_value_chars - 3)].rstrip() + "..."
        return text

    def _evidence_priority(self, view: str, item: dict[str, Any]) -> int:
        value = str(item.get("value", "") or "")
        lower = value.lower()
        upper = value.upper()
        score = 0

        for rule in BEHAVIOR_RULES:
            keywords = rule.get("keywords", {}).get(view, [])
            for keyword in keywords:
                if str(keyword).lower() in lower:
                    score += 12

        if view == "permission":
            dangerous_tokens = (
                "SEND_SMS",
                "READ_SMS",
                "RECEIVE_SMS",
                "CALL_PHONE",
                "READ_PHONE_STATE",
                "READ_CONTACTS",
                "GET_ACCOUNTS",
                "ACCESS_FINE_LOCATION",
                "SYSTEM_ALERT_WINDOW",
                "BIND_ACCESSIBILITY_SERVICE",
                "REQUEST_INSTALL_PACKAGES",
                "RECEIVE_BOOT_COMPLETED",
                "WRITE_EXTERNAL_STORAGE",
                "MANAGE_EXTERNAL_STORAGE",
            )
            if any(token in upper for token in dangerous_tokens):
                score += 18
            if "INTERNET" in upper or "ACCESS_NETWORK_STATE" in upper:
                score += 8
        elif view == "api":
            api_markers = (
                "dexclassloader",
                "classloader",
                "runtime;->exec",
                "processbuilder",
                "smsmanager",
                "telephonymanager",
                "locationmanager",
                "webview;->loadurl",
                "packageinstaller",
                "system;->load",
                "loadlibrary",
                "cipher",
                "messagedigest",
                "base64",
                "urlconnection",
                "socket",
            )
            if any(marker in lower for marker in api_markers):
                score += 18
            if "android/" in lower or "java/" in lower or "javax/" in lower:
                score += 2
        elif view == "string":
            if any(marker.lower() in lower for marker in SUSPICIOUS_STRING_MARKERS):
                score += 18
            if URL_RE.search(value):
                score += 10
            if any(marker in lower for marker in ("payload", "classes.dex", "content://sms", "su", "chmod", ".apk", ".so")):
                score += 10
            if len(value) > 120:
                score -= 2
        elif view == "component":
            if any(marker in lower for marker in ("receiver", "service", "provider", "boot", "sms", "accessibility", "admin")):
                score += 10
            detail = item.get("detail", {})
            if isinstance(detail, dict):
                if detail.get("exported") is True:
                    score += 5
                component_type = str(detail.get("type", "") or "").lower()
                if component_type in {"receiver", "service", "provider"}:
                    score += 5

        return score

    def _validate(self, evidence_doc: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        index = evidence_by_id(evidence_doc)
        allowed = {rule["label"]: rule for rule in BEHAVIOR_RULES}
        behaviors: list[dict[str, Any]] = []
        for item in payload.get("behaviors", []):
            label = item.get("label")
            if label not in allowed:
                continue
            evidence_ids = []
            for evidence_id in item.get("evidence_ids", []):
                if evidence_id in index and evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)
            if not evidence_ids:
                continue
            rule = allowed[label]
            behavior = {
                "label": label,
                "name": item.get("name") or rule["name"],
                "description": item.get("description") or rule["description"],
                "evidence_ids": evidence_ids,
                "support_by_view": support_by_view(evidence_ids, index),
                "consistency_score": consistency_score(evidence_ids, index),
                "analyzer": "llm",
            }
            malware_relevance = _optional_unit_float(item.get("malware_relevance"))
            confidence = _optional_unit_float(item.get("confidence"))
            risk_level = _risk_level(item.get("risk_level"))
            if malware_relevance is not None:
                behavior["malware_relevance"] = malware_relevance
            if confidence is not None:
                behavior["confidence"] = confidence
            if risk_level:
                behavior["risk_level"] = risk_level
            behaviors.append(behavior)
        behaviors.sort(key=lambda row: row["consistency_score"], reverse=True)
        result: dict[str, Any] = {
            "sample_id": evidence_doc["sample_id"],
            "label": evidence_doc.get("label"),
            "analyzer": f"llm:{self.model}",
            "behaviors": behaviors,
        }
        risk_score = _optional_unit_float(payload.get("apk_risk_score"))
        risk_level = _risk_level(payload.get("risk_level"))
        risk_rationale = str(payload.get("risk_rationale", "") or "").strip()
        if risk_score is not None or risk_level or risk_rationale:
            result["llm_risk"] = {}
            if risk_score is not None:
                result["llm_risk"]["apk_risk_score"] = risk_score
            if risk_level:
                result["llm_risk"]["risk_level"] = risk_level
            if risk_rationale:
                result["llm_risk"]["risk_rationale"] = risk_rationale[:500]
        return result


def _thinking_type(value: dict[str, Any] | str | bool | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"enabled", "disabled"}:
            return lowered
        if lowered in {"true", "yes", "on"}:
            return "enabled"
        if lowered in {"false", "no", "off"}:
            return "disabled"
    if isinstance(value, dict):
        raw_type = str(value.get("type", "")).strip().lower()
        if raw_type in {"enabled", "disabled"}:
            return raw_type
    raise ValueError(f"Unsupported thinking config: {value!r}")


def _loads_json_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if not content:
        raise ValueError("LLM returned empty content.")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(content[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON output must be an object.")
    return payload


def _optional_unit_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, parsed))


def _risk_level(value: Any) -> str:
    level = str(value or "").strip().lower()
    return level if level in {"low", "medium", "high", "critical"} else ""


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_view_budgets(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("view_budgets must be a JSON object/dict")
    budgets: dict[str, int] = {}
    for view, budget in value.items():
        parsed = _optional_int(budget)
        if parsed is None:
            continue
        budgets[str(view)] = max(0, parsed)
    return budgets


def _strip_url_noise(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        scheme = raw.split("://", 1)[0] if "://" in raw else "http"
        host = match.group(1) or ""
        path = match.group(2) or ""
        if len(path) > 48:
            path = path[:45] + "..."
        return f"{scheme}://{host}{path}"

    return URL_RE.sub(repl, value)


def _normalized_signature(view: str, value: str) -> str:
    text = SPACE_RE.sub(" ", value).strip().lower()
    if view == "string":
        text = _strip_url_noise(text)
    text = re.sub(r"\d+", "#", text)
    return text[:240]
